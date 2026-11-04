"""
cox_model_api.py
COX activity classifier (Random Forest on ECFP4, 2048 bits).

Needs: rdkit, scikit-learn, joblib, numpy, pandas
Files that must sit in the same folder:
    cox_model.joblib        trained model
    train_smiles_cox.csv    training SMILES (column "smiles"), for the reliability flag

Usage:
    import cox_model_api as cox
    X     = cox.featurize(smiles_list)        # feature matrix for TreeSHAP
    model = cox.load_model()                  # sklearn RandomForestClassifier
    df    = cox.predict(smiles_list)          # proba, label, reliability
    names = cox.bit_names(smiles_list)        # readable names for the 2048 features
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
MODEL_FILE = HERE / "cox_model.joblib"
TRAIN_FILE = HERE / "train_smiles_cox.csv"

# fingerprint settings must match training exactly
RADIUS, N_BITS = 2, 2048
CUTOFF = 0.5  # <-- replace with the tuned cutoff if you have one

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
_model = None
_train_fps = None


def load_model():
    """Returns the trained RandomForestClassifier (cached)."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_FILE)
    return _model


def _fp(smiles):
    """SMILES -> RDKit fingerprint object, or None if unparsable."""
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else _fp_gen.GetFingerprint(mol)


def featurize(smiles_list):
    """
    SMILES -> numpy array (n_molecules x 2048), 0/1 bits.
    This is the X you pass to shap.TreeExplainer(model).shap_values(X).
    Unparsable SMILES become all-zero rows, so the row order is preserved.
    """
    rows = []
    for s in smiles_list:
        arr = np.zeros((N_BITS,), dtype=np.uint8)
        fp = _fp(s)
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.array(rows)


def _load_train_fps():
    global _train_fps
    if _train_fps is None:
        smis = pd.read_csv(TRAIN_FILE)["smiles"]
        _train_fps = [f for f in (_fp(s) for s in smis) if f is not None]
    return _train_fps


def predict(smiles_list, cutoff=CUTOFF, with_reliability=True):
    """
    Returns a DataFrame with:
        smiles, proba, active (0/1), max_sim_train, in_domain
    proba is NaN for unparsable SMILES.
    max_sim_train = highest Tanimoto similarity to any training compound.
    The model is clearly weaker below ~0.5, so treat those predictions with care.
    """
    model = load_model()
    X = featurize(smiles_list)
    valid = np.array([_fp(s) is not None for s in smiles_list])

    proba = np.full(len(smiles_list), np.nan)
    if valid.any():
        proba[valid] = model.predict_proba(X[valid])[:, 1]

    out = pd.DataFrame(
        {
            "smiles": smiles_list,
            "proba": proba,
            "active": np.where(
                np.isnan(proba), np.nan, (proba >= cutoff).astype(float)
            ),
        }
    )

    if with_reliability:
        ref = _load_train_fps()
        sims = []
        for s in smiles_list:
            fp = _fp(s)
            sims.append(
                np.nan
                if fp is None
                else max(DataStructs.BulkTanimotoSimilarity(fp, ref))
            )
        out["max_sim_train"] = sims
        out["in_domain"] = out["max_sim_train"] >= 0.5

    return out


def bit_names(smiles_list, prefix="bit"):
    """
    Readable names for the 2048 features, for SHAP plots.
    For every bit that is set in the given molecules, the name carries the SMARTS
    of one substructure that switches that bit on, e.g. "bit314 [c]1ccccc1".
    Bits never set keep their plain name.
    """
    names = [f"{prefix}{i}" for i in range(N_BITS)]
    seen = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        info = rdFingerprintGenerator.AdditionalOutput()
        info.AllocateBitInfoMap()
        _fp_gen.GetFingerprint(mol, additionalOutput=info)
        for bit, envs in info.GetBitInfoMap().items():
            if bit in seen:
                continue
            atom, rad = envs[0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom) if rad else None
            frag = (
                Chem.MolFragmentToSmiles(mol, atomsToUse=[atom])
                if env is None
                else Chem.MolToSmiles(Chem.PathToSubmol(mol, env))
            )
            names[bit] = f"{prefix}{bit} {frag}"
            seen.add(bit)
    return names
