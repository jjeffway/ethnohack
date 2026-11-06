"""
cox_preprocess.py
Step 1 of 2: raw data -> clean, scaffolded, featurised table.

Run this whenever the raw data or the feature set changes.
Writes COX2_PREPARED_CSV, which cox_train.py reads.

Needs: rdkit, numpy, pandas

Usage (notebook):
    import ethnohack.cox_model.cox_preprocess as prep
    data = prep.build_dataset()          # does everything, writes the CSV
    data.head()
"""

import numpy as np
import pandas as pd
from ethnohack.constants import (
    COX2_CSV,
    COX2_PREPARED_CSV,
    FEATURE_MODE,
    FP_SIZE,
    MORGAN_RADIUS,
)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
    rdReducedGraphs,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Columns that describe the molecule rather than being model input.
# Everything else numeric in the prepared table is treated as a feature.
META_COLS = {"smiles", "label", "scaffold", "inchikey", "name", "source", "id"}

# 10 simple physicochemical properties
DESC = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumRings", rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
]

_MACCS_LEN = 167  # 166 keys + unused bit 0
# ErG length depends on RDKit's default path settings, so measure it once
_ERG_LEN = len(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles("c1ccccc1")))


# ============================================================
# 1) CLEANING
# ============================================================
def load_raw(path=COX2_CSV):
    return pd.read_csv(path)


def clean_data(df, smiles_col="Smiles", label_col="Activity", keep_cols=None):
    """
    Returns a DataFrame with smiles, label, plus any columns in keep_cols
    (use this to carry your friend's extra descriptor columns through).
    SMILES are assumed canonical already.
    """
    keep_cols = list(keep_cols or [])
    data = df[[smiles_col, label_col] + keep_cols].copy()
    data = data.rename(columns={smiles_col: "smiles", label_col: "label"})
    data["label"] = data["label"].astype(int)
    data = data.dropna(subset=["smiles"])

    ok = data["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    print("Unparsable SMILES removed:", int((~ok).sum()))
    data = data[ok]

    # same molecule labelled both 0 and 1 -> untrustworthy, drop it entirely
    conflict = data.groupby("smiles")["label"].nunique()
    data = data[~data["smiles"].isin(conflict[conflict > 1].index)]
    data = data.drop_duplicates(subset="smiles").reset_index(drop=True)

    print(
        f"Compounds: {len(data)} | actives: {data['label'].sum()} "
        f"({data['label'].mean():.1%})"
    )
    return data


def scaffold_of(smi):
    """Murcko scaffold = the ring core, side chains stripped off."""
    mol = Chem.MolFromSmiles(smi)
    scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return scaf if scaf != "" else "ACYCLIC_" + smi  # ring-free: own group


def add_scaffolds(data):
    """Adds the scaffold column used to group molecules for the honest split."""
    data = data.copy()
    data["scaffold"] = data["smiles"].apply(scaffold_of)
    print(f"Unique scaffolds: {data['scaffold'].nunique()}")
    return data


# ============================================================
# 2) FEATURES
#    This is the ONE place features are defined. Training and prediction
#    both call compute_rdkit_features, so they can never disagree.
#
#    Modes:
#      ecfp           2048 fingerprint bits
#      ecfp_counts    same, but counts instead of 0/1
#      ecfp_desc      fingerprint + 10 properties
#      ecfp_extended  fingerprint + ErG + MACCS + 10 properties
# ============================================================
def feature_spec(mode=FEATURE_MODE, fp_size=FP_SIZE):
    """Returns the column names this mode produces, in order."""
    names = [f"fp_{i}" for i in range(fp_size)]
    if mode == "ecfp_desc":
        names += [n for n, _ in DESC]
    elif mode == "ecfp_extended":
        names += [f"ErG_{i}" for i in range(_ERG_LEN)]
        names += [f"MACCS_{i}" for i in range(_MACCS_LEN)]
        names += [n for n, _ in DESC]
    return names


def compute_rdkit_features(
    smiles_list, mode=FEATURE_MODE, fp_size=FP_SIZE, radius=MORGAN_RADIUS
):
    """
    SMILES -> DataFrame of features (one row per molecule, order preserved).
    Unparsable SMILES give all-zero rows.

    >>> ADD NEW FEATURES HERE <<<
    Anything added must be computable from the SMILES alone, otherwise it
    cannot be produced for the plant compounds at prediction time.
    Remember to add the matching names to feature_spec().
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    names = feature_spec(mode, fp_size)
    n_cols = len(names)

    rows = []
    for s in smiles_list:
        vec = np.zeros((n_cols,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            if mode == "ecfp_counts":
                # counts matter when a substructure appearing 3x is meaningful
                vec[:fp_size] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr = np.zeros((fp_size,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
                vec[:fp_size] = arr

            extra = []
            if mode == "ecfp_extended":
                # ErG = pharmacophore pattern, built for scaffold hopping:
                # it may transfer better to unfamiliar plant chemistry
                extra.append(
                    np.array(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
                )
                maccs = np.zeros((_MACCS_LEN,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
                extra.append(maccs.astype(np.float32))
            if mode in ("ecfp_desc", "ecfp_extended"):
                extra.append(np.array([f(mol) for _, f in DESC], dtype=np.float32))

            if extra:
                vec[fp_size:] = np.concatenate(extra)
        rows.append(vec)

    X = np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(X, columns=names)


def feature_columns(data):
    """All numeric columns that are not metadata = the model's input columns."""
    return [
        c
        for c in data.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(data[c])
    ]


# if you need to add some columns you can do it with this: prep.build_dataset(keep_cols=["her_descriptor_1", "her_descriptor_2"]) in the notebook when your calling on it if they are descriptors that you got externally, if it is from rdkit you can add it internally as well you can add them to DESC in the beginning


# ============================================================
# 3) BUILD AND SAVE
# ============================================================
def build_dataset(
    path=COX2_CSV,
    smiles_col="Smiles",
    label_col="Activity",
    keep_cols=None,
    mode=FEATURE_MODE,
    save=True,
):
    """
    Full preprocessing: load -> clean -> scaffolds -> features -> save.
    keep_cols: extra columns from the raw file to carry through as features.
    """
    data = add_scaffolds(clean_data(load_raw(path), smiles_col, label_col, keep_cols))

    feats = compute_rdkit_features(data["smiles"], mode)
    feats.index = data.index
    data = pd.concat([data, feats], axis=1)

    print(f"Feature columns: {len(feature_columns(data))}  (mode: {mode})")

    if save:
        COX2_PREPARED_CSV.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(COX2_PREPARED_CSV, index=False)
        print(f"Saved: {COX2_PREPARED_CSV}")
    return data
