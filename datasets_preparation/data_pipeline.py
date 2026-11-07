"""
Pipeline script for processing ChEMBL data.

This script performs the following steps:
1. Clean raw CSV: filter columns, keep rows with pChEMBL Value, group by molecule, binarize activity
2. Compute RDKit molecular features
3. Prune redundant features based on correlation
4. Build final model table with selected features

Usage:
    python data_pipeline.py raw_data.csv [--output OUTPUT_CSV]

Arguments:
    raw_data.csv    Path to the raw CSV file (e.g., 5lox.csv or cox2db.csv)
    --output, -o    Output CSV filename (default: {input_name}_model_table.csv)

Output:
    Creates a CSV file with the final processed data containing:
    - ChEMBL_IDs: Semicolon-separated list of ChEMBL IDs (for merged stereoisomers)
    - scaffold: Murcko scaffold for each molecule
    - Activity: Binary activity (1 for active, 0 for inactive)
    - Molecular features computed by RDKit

Note:
    The script requires pandas, numpy, rdkit, and scipy to be installed.
    It processes all rows with non-null pChEMBL Value, regardless of Standard Type.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# =============================================================================
# Configuration
# =============================================================================

# Step 1: Cleaning
CLEAN_COLUMNS = [
    "Molecule ChEMBL ID",
    "Compound Key",
    "Smiles",
    "Standard Type",
    "pChEMBL Value",
]
BINARY_THRESHOLD = 6.0
ID_COL = "Molecule ChEMBL ID"
SCAFFOLD_COL = "scaffold"
LABEL_COL = "Activity"

# Step 2: Feature computation
CORE_FEATURES = {
    "MolWt": Descriptors.MolWt,  # pyright: ignore[reportAttributeAccessIssue]
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "Kappa1": Descriptors.Kappa1,  # pyright: ignore[reportAttributeAccessIssue]
    "Kappa2": Descriptors.Kappa2,  # pyright: ignore[reportAttributeAccessIssue]
    "Kappa3": Descriptors.Kappa3,  # pyright: ignore[reportAttributeAccessIssue]
    "MolLogP": Descriptors.MolLogP,  # pyright: ignore[reportAttributeAccessIssue]
}

# Build full feature function dictionary
_REGISTERED = dict(Descriptors._descList)
FEATURE_FUNCS = {**CORE_FEATURES}
for name, func in _REGISTERED.items():
    if name.startswith(("Chi", "EState_VSA", "fr_")):
        FEATURE_FUNCS.setdefault(name, func)

# Step 3: Pruning
PRUNING_THRESHOLD = 0.5  # |r| above which two features count as redundant
PRUNING_TIE_MARGIN = 0.02  # redundancy scores closer than this are treated as a tie
PROTECTED_FEATURES = {"MolWt", "FractionCSP3", "TPSA"}

# Step 4: Final table
STEREO_MODE = "merge"  # "merge" or "keep"
AGGREGATE = "median"  # how to combine labels of merged stereoisomers

# Standardization tools
normalizer = rdMolStandardize.Normalizer()
uncharger = rdMolStandardize.Uncharger()
lfc = rdMolStandardize.LargestFragmentChooser()

# =============================================================================
# Helper Functions
# =============================================================================


def standardize(smi):
    """Standardize a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = lfc.choose(mol)
    mol = normalizer.normalize(mol)
    mol = uncharger.uncharge(mol)
    return mol


def compute_all_features(mol):
    """Compute all features for a molecule."""
    if mol is None:
        return None
    feats = {name: func(mol) for name, func in FEATURE_FUNCS.items()}
    return feats


def compute_features_for_list(mol, feature_list):
    """Compute only the specified features for a molecule."""
    if mol is None:
        return None
    feats = {f: FEATURE_FUNCS[f](mol) for f in feature_list if f in FEATURE_FUNCS}
    return feats


# =============================================================================
# Step 1: Clean raw data
# =============================================================================


def clean_raw_data(raw_df):
    """Clean the raw CSV data."""
    print("Step 1: Cleaning raw data...")

    # Select relevant columns
    cols = [c for c in CLEAN_COLUMNS if c in raw_df.columns]
    df = raw_df[cols].copy()

    # Only keep rows with non-null pChEMBL Value
    if "pChEMBL Value" in df.columns:
        df = df[df["pChEMBL Value"].notna()]

    # Group by molecule and aggregate
    if "Molecule ChEMBL ID" in df.columns and "pChEMBL Value" in df.columns:
        df = (
            df.groupby("Molecule ChEMBL ID", as_index=True)
            .agg(
                {
                    "Smiles": "first",
                    "pChEMBL Value": "median",
                }
            )
            .reset_index()
        )

    # Sort by activity value
    if "pChEMBL Value" in df.columns:
        df = df.sort_values(by="pChEMBL Value", ascending=False)

    # Create binary activity column
    df_binary = df.dropna(subset=["pChEMBL Value", "Smiles"]).copy()
    if "pChEMBL Value" in df_binary.columns:
        df_binary[LABEL_COL] = np.where(
            df_binary["pChEMBL Value"] >= BINARY_THRESHOLD, 1, 0
        )

    # Add scaffold column
    if "Smiles" in df_binary.columns:
        df_binary[SCAFFOLD_COL] = df_binary["Smiles"].apply(
            lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        )

    print(f"  Input: {len(raw_df)} rows")
    print(f"  After filtering: {len(df)} molecules")
    print(f"  After binarization: {len(df_binary)} molecules")
    if LABEL_COL in df_binary.columns:
        n_active = (df_binary[LABEL_COL] == 1).sum()
        n_inactive = (df_binary[LABEL_COL] == 0).sum()
        print(f"  Active: {n_active}, Inactive: {n_inactive}")

    return df_binary


# =============================================================================
# Step 2: Compute features
# =============================================================================


def compute_features(df_clean):
    """Compute all RDKit features for cleaned data."""
    print("\nStep 2: Computing features...")

    records = []
    skipped = 0

    for idx, row in df_clean.iterrows():
        mol = standardize(row["Smiles"])
        if mol is None:
            skipped += 1
            continue

        feats = compute_all_features(mol)
        if feats is None:
            skipped += 1
            continue

        records.append(feats)

    features = pd.DataFrame(records, index=df_clean.index)

    # Drop features with all NaN or no variation
    features = features.dropna(axis=1)
    features = features.loc[:, features.nunique() > 1]

    print(f"  Computed {features.shape[1]} features for {features.shape[0]} molecules")
    print(f"  Skipped {skipped} molecules")

    return features


# =============================================================================
# Step 3: Prune features
# =============================================================================


def prune_features(features, activity):
    """Prune redundant features based on correlation."""
    print("\nStep 3: Pruning features...")

    corr = features.corr(method="spearman")
    act_r = features.apply(lambda c: c.corr(activity, method="spearman")).fillna(0.0)

    # Find all redundant pairs
    pairs = sorted(
        (
            (abs(corr.loc[a, b]), a, b)
            for a, b in itertools.combinations(corr.columns, 2)
            if abs(corr.loc[a, b]) > PRUNING_THRESHOLD
        ),
        reverse=True,
    )

    def redundancy(feature, pool, corr):
        """Mean absolute correlation of a feature to the other remaining features."""
        others = [c for c in pool if c != feature]
        return corr.loc[feature, others].abs().mean()

    remaining = set(corr.columns)
    log = []

    for r, a, b in pairs:
        if a not in remaining or b not in remaining:
            continue
        if a in PROTECTED_FEATURES and b in PROTECTED_FEATURES:
            continue

        if a in PROTECTED_FEATURES:
            drop, keep, rule = b, a, "protected"
        elif b in PROTECTED_FEATURES:
            drop, keep, rule = a, b, "protected"
        else:
            ra = redundancy(a, remaining, corr)
            rb = redundancy(b, remaining, corr)
            if abs(ra - rb) > PRUNING_TIE_MARGIN:
                drop, keep = (a, b) if ra > rb else (b, a)
                rule = "redundancy"
            else:
                drop, keep = (a, b) if abs(act_r[a]) < abs(act_r[b]) else (b, a)
                rule = "activity tie-break"

        remaining.discard(drop)
        log.append(
            {
                "dropped": drop,
                "kept": keep,
                "pair_r": round(corr.loc[a, b], 3),
                "rule": rule,
                "dropped_activity_r": round(act_r[drop], 3),
                "kept_activity_r": round(act_r[keep], 3),
            }
        )

    print(f"  Dropped {len(log)} features, kept {len(remaining)}")

    return sorted(remaining)


# =============================================================================
# Step 4: Build final model table
# =============================================================================


def build_final_table(df_clean, selected_features):
    """Build the final model table with selected features."""
    print("\nStep 4: Building final model table...")

    unknown = [f for f in selected_features if f not in FEATURE_FUNCS]
    if unknown:
        raise ValueError(f"No function defined for: {unknown}")

    df = df_clean.copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        mol = standardize(row["Smiles"])
        if mol is None or pd.isna(row[LABEL_COL]):
            skipped += 1
            continue

        stereo_smi = Chem.MolToSmiles(mol)
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        flat_smi = Chem.MolToSmiles(flat)

        feats = compute_features_for_list(flat, selected_features)
        if feats is None:
            skipped += 1
            continue

        records.append(
            {
                "ChEMBL_ID": row[ID_COL],
                "flat_smiles": flat_smi,
                "stereo_smiles": stereo_smi,
                SCAFFOLD_COL: row[SCAFFOLD_COL],
                LABEL_COL: row[LABEL_COL],
                **feats,
            }
        )

    table = pd.DataFrame(records)
    print(f"  Usable rows: {len(table)}, skipped: {skipped}")
    print(f"  Unique flat structures: {table['flat_smiles'].nunique()}")

    # Handle stereoisomers
    if STEREO_MODE == "merge":
        # Log stereoisomers
        multi = table[table.groupby("flat_smiles")["flat_smiles"].transform("size") > 1]
        if not multi.empty:
            print(f"  {multi['flat_smiles'].nunique()} structures have stereoisomers")

        # Merge stereoisomers
        feature_cols = [f for f in selected_features if f in table.columns]
        merged = (
            table.groupby("flat_smiles")
            .agg(
                ChEMBL_IDs=("ChEMBL_ID", lambda s: ";".join(map(str, s))),
                Smiles=("flat_smiles", "first"),
                scaffold=(SCAFFOLD_COL, "first"),
                **{LABEL_COL: (LABEL_COL, AGGREGATE)},
                **{f: (f, "first") for f in feature_cols},
            )
            .reset_index(drop=True)
        )

        final = merged
    elif STEREO_MODE == "keep":
        final = table.rename(columns={"ChEMBL_ID": "ChEMBL_IDs"})
        feature_cols = [f for f in selected_features if f in table.columns]
        final = final[
            ["ChEMBL_IDs", "flat_smiles", SCAFFOLD_COL, LABEL_COL] + feature_cols
        ]
        final = final.rename(columns={"flat_smiles": "Smiles"})
    else:
        raise ValueError("STEREO_MODE must be 'merge' or 'keep'")

    print(f"  Final table: {final.shape[0]} rows x {final.shape[1]} columns")

    return final


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Process ChEMBL data through the full pipeline"
    )
    parser.add_argument(
        "input_csv", help="Path to the raw CSV file (e.g., 5lox.csv or cox2db.csv)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV filename (default: {input_name}_model_table.csv)",
        default=None,
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (cleaned CSV, features CSV, selected features CSV)",
    )

    args = parser.parse_args()

    # Read input
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Processing: {input_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_model_table.csv"

    # Read raw data
    try:
        raw_df = pd.read_csv(input_path, sep=None, engine="python", on_bad_lines="warn")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    # Step 1: Clean raw data
    df_clean = clean_raw_data(raw_df)

    # Step 2: Compute features
    features = compute_features(df_clean)

    # Step 3: Prune features
    # Need to align features with cleaned data indices
    features_aligned = features.reindex(df_clean.index)
    activity = df_clean[LABEL_COL].reindex(features_aligned.index)

    selected_features = prune_features(features_aligned, activity)

    # Step 4: Build final table
    final_table = build_final_table(df_clean, selected_features)

    # Save final output
    final_table.to_csv(output_path, index=False)
    print(f"\nFinal output saved to: {output_path}")
    print(f"  Shape: {final_table.shape[0]} rows x {final_table.shape[1]} columns")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
