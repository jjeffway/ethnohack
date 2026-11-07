"""
cox_preprocess.py
Step 1 of 2: raw table -> clean, scaffolded, model-ready table.

IMPORTANT: the raw table already carries the descriptor columns, so NO
features are computed during training. compute_rdkit_features() exists for
PREDICTION, where the same columns must be derived from a plant SMILES.

Needs: rdkit, numpy, pandas

Usage (notebook):
    import ethnohack.cox_model.cox_preprocess as prep

    data = prep.build_dataset()        # does everything, writes the CSV
    prep.verify_features(data)         # only possible if a SMILES column exists
"""

import numpy as np
import pandas as pd
from ethnohack.constants import COX2_CSV, COX2_PREPARED_CSV
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Columns that identify or label a molecule rather than describing it.
# Everything else numeric in the prepared table becomes a model feature.
# "Activity" MUST be in here: it is the label, and if it leaked into the
# feature list the model would score near 1.0 and mean nothing.
META_COLS = {
    "smiles", "Smiles", "canonical_smiles", "flat",
    "label", "Activity", "activity",
    "scaffold", "ChEMBL_IDs", "Molecule ChEMBL ID",
    "inchikey", "name", "source", "id",
}


# ============================================================
# 1) LOADING AND CLEANING
# ============================================================
def load_raw(path=COX2_CSV):
    """Reads the raw CSV as it comes from the feature-engineering step."""
    df = pd.read_csv(path)
    print(f"Raw: {len(df)} rows, {len(df.columns)} columns")
    return df


def find_column(df, candidates, required=True, what="column"):
    """Finds a column by any of several possible names, case-insensitive."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"No {what} found. Tried: {candidates}")
    return None


def clean_data(df, label_col=None, smiles_col=None):
    """
    Returns a table with a 'label' column (0/1) plus every other column
    carried through unchanged.

    Renaming the label to 'label' is not cosmetic: feature_columns() excludes
    'label', so a column still called 'Activity' would be treated as a feature.
    """
    data = df.copy()

    label_col = label_col or find_column(
        data, ["Activity", "activity", "label"], what="label column"
    )
    if label_col != "label":
        data = data.rename(columns={label_col: "label"})
        print(f"Renamed label column: {label_col!r} -> 'label'")

    data["label"] = data["label"].astype(int)

    # a SMILES column is optional here, but it enables the similarity checks,
    # the reliability bins and verify_features()
    smiles_col = smiles_col or find_column(
        data, ["smiles", "canonical_smiles"], required=False, what="SMILES column"
    )
    if smiles_col:
        if smiles_col != "smiles":
            data = data.rename(columns={smiles_col: "smiles"})
        ok = data["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        if (~ok).any():
            print(f"Unparsable SMILES removed: {int((~ok).sum())}")
            data = data[ok]
    else:
        print("WARNING: no SMILES column. Prediction still works, but the "
              "similarity baseline, the reliability bins and verify_features() "
              "cannot run.")

    data = data.reset_index(drop=True)
    print(f"Compounds: {len(data)} | actives: {data['label'].sum()} "
          f"({data['label'].mean():.1%})")
    return data


def remove_feature_duplicates(data):
    """
    Drops rows that are indistinguishable to the model.

    Why this matters here: the descriptor columns are stereo-blind. MolWt,
    TPSA, the fr_* fragment counts and the EState_VSA columns all give
    identical values for two enantiomers. So stereoisomers become duplicate
    rows in feature space, and where they carry different labels the model
    cannot resolve them in principle. Such rows also leak across a scaffold
    split unnoticed, because the scaffold strings may differ while the
    features do not.

    Note this discards real chemistry: COX inhibition by profens is
    stereoselective. The representation simply cannot express it, so those
    rows are noise rather than signal.
    """
    cols = feature_columns(data)
    if not cols:
        print("No feature columns yet, skipping duplicate removal")
        return data

    before = len(data)

    # hash each feature row so identical vectors group together;
    # rounding absorbs tiny floating-point differences
    key = pd.util.hash_pandas_object(data[cols].round(6), index=False)
    data = data.assign(_key=key)

    # 1) groups whose rows disagree about the label
    conflict = data.groupby("_key")["label"].nunique()
    bad = set(conflict[conflict > 1].index)
    n_conflict_rows = int(data["_key"].isin(bad).sum())
    data = data[~data["_key"].isin(bad)]

    # 2) the remaining duplicates agree on the label, so keep one of each
    n_dupes = int(data["_key"].duplicated().sum())
    data = data.drop_duplicates(subset="_key").drop(columns="_key")
    data = data.reset_index(drop=True)

    print(f"Conflicting feature vectors: {len(bad)} groups "
          f"({n_conflict_rows} rows) removed")
    print(f"Duplicate feature vectors:   {n_dupes} rows removed")
    print(f"{before} -> {len(data)} compounds ({data['label'].mean():.1%} active)")
    return data


# ============================================================
# 2) SCAFFOLDS
#    Used to group analogues so they cannot land on both sides of the split.
# ============================================================
def scaffold_of(smi):
    """Murcko scaffold = the ring core, side chains stripped off."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return scaf if scaf != "" else "ACYCLIC_" + str(smi)   # ring-free: own group


def add_scaffolds(data):
    """
    Ensures a usable scaffold column. If the raw table already has one it is
    kept; otherwise it is computed from the SMILES.
    """
    data = data.copy()

    if "scaffold" in data.columns and data["scaffold"].notna().all():
        print(f"Scaffold column already present: {data['scaffold'].nunique()} unique")
        return data

    if "smiles" not in data.columns:
        raise KeyError(
            "No scaffold column and no SMILES to compute one from. "
            "The scaffold split needs one of the two."
        )

    data["scaffold"] = data["smiles"].apply(scaffold_of)
    missing = data["scaffold"].isna().sum()
    if missing:
        print(f"Dropping {missing} rows without a scaffold")
        data = data.dropna(subset=["scaffold"]).reset_index(drop=True)
    print(f"Unique scaffolds: {data['scaffold'].nunique()}")
    return data


# ============================================================
# 3) FEATURES
#    Training reads the descriptor columns straight from the table.
#    compute_rdkit_features() is the PREDICTION path: it must be able to
#    reproduce those same columns from a SMILES string alone.
# ============================================================
def feature_columns(data):
    """All numeric columns that are not metadata = the model's input columns."""
    return [
        c for c in data.columns
        if c not in META_COLS
        and not c.startswith("_")
        and pd.api.types.is_numeric_dtype(data[c])
    ]


def compute_rdkit_features(smiles_list):
    """
    SMILES -> DataFrame of ALL RDKit descriptors (~210 columns).

    Deliberately a superset: featurize() in the API selects the subset the
    model was trained on, so this does not need to know which columns were
    kept during feature selection.

    Anything added here must be computable from a SMILES alone, otherwise it
    cannot be produced for the plant compounds at prediction time.
    Unparsable SMILES give all-zero rows, so row order is preserved.
    """
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        rows.append({} if mol is None else Descriptors.CalcMolDescriptors(mol))
    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def verify_features(data, n=50, tol=1e-3):
    """
    Checks that compute_rdkit_features() reproduces the table's descriptor
    columns. This is the single most important check in the pipeline: training
    reads columns from a file while prediction computes them, and if the two
    disagree the model silently receives wrong input on the plant compounds.

    Needs a SMILES column. Returns True when the two paths agree.
    """
    if "smiles" not in data.columns:
        print("Cannot verify: no SMILES column in the table.")
        print("You are trusting that the descriptors were computed with plain "
              "RDKit defaults and not scaled or normalised afterwards.")
        return None

    cols = feature_columns(data)
    sample = data.head(n)
    recomputed = compute_rdkit_features(sample["smiles"])

    missing = sorted(set(cols) - set(recomputed.columns))
    if missing:
        print(f"NOT reproducible by RDKit ({len(missing)}): {missing[:10]}")
        print("Prediction on new molecules will fail for these columns.")
        return False

    a = sample[cols].to_numpy(dtype=float)
    b = recomputed[cols].to_numpy(dtype=float)
    diff = np.nanmax(np.abs(a - b))
    print(f"All {len(cols)} columns reproducible | largest difference: {diff:.6g}")

    if diff > tol:
        worst = np.nanmax(np.abs(a - b), axis=0)
        bad = [cols[i] for i in np.argsort(-worst)[:5]]
        print(f"MISMATCH above {tol}. Worst columns: {bad}")
        print("The descriptors were probably scaled, normalised or computed "
              "with a different tool. Resolve this before predicting.")
        return False

    print("Training and prediction paths agree.")
    return True


# ============================================================
# 4) BUILD AND SAVE
# ============================================================
def build_dataset(path=COX2_CSV, label_col=None, smiles_col=None,
                  dedupe=True, save=True):
    """
    Full preprocessing: load -> clean -> dedupe -> scaffolds -> save.
    No features are computed; they already sit in the raw table.
    """
    data = clean_data(load_raw(path), label_col, smiles_col)

    if dedupe:
        data = remove_feature_duplicates(data)

    data = add_scaffolds(data)

    cols = feature_columns(data)
    print(f"\nFeature columns: {len(cols)}")
    print(f"First few: {cols[:8]}")

    # guard against the worst possible mistake in this pipeline
    assert "label" not in cols, "the label leaked into the feature list!"
    for name in ("Activity", "activity"):
        assert name not in cols, f"{name!r} leaked into the feature list!"

    if save:
        COX2_PREPARED_CSV.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(COX2_PREPARED_CSV, index=False)
        print(f"Saved: {COX2_PREPARED_CSV}")
    return data
