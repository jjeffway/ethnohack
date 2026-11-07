"""
cox_model_api.py
Loading and using the trained COX-2 classifier. No training code and no
feature definitions here — features come from cox_preprocess so that
prediction can never disagree with how the model was trained.

Pipeline: cox_preprocess.py -> cox_train.py -> this file.

The model trains on descriptor columns read from a prepared table. This file
reproduces those same columns from a SMILES string, which is how the plant
compounds get scored.

Needs: rdkit, scikit-learn, joblib, numpy, pandas

Usage:
    import ethnohack.cox_model.cox_model_api as cox

    cox.model_info()                    # what was loaded, and how it scored
    df    = cox.predict(smiles_list)    # proba, active, reliability
    X     = cox.featurize(smiles_list)  # feature matrix for TreeSHAP
    names = cox.feature_names()         # column names, same order as X
    model = cox.load_model()            # sklearn RandomForestClassifier

TreeSHAP example:
    import shap
    X      = cox.featurize(smiles_list)
    sv     = shap.TreeExplainer(cox.load_model()).shap_values(X)
    sv_act = sv[1] if isinstance(sv, list) else sv[..., 1]   # class "active"
    shap.summary_plot(sv_act, X, feature_names=cox.feature_names())

    # Filter to in_domain == True first where that column is available.
    # Explaining predictions the model cannot make reliably produces
    # confident-looking but meaningless rationales.
"""

import joblib
import numpy as np
import pandas as pd
from ethnohack.constants import COX2_MODEL_FILE, COX2_TRAIN_FILE, SIM_CUTOFF
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# the single definition of how molecules become features
from .cox_preprocess import compute_rdkit_features

RDLogger.DisableLog("rdApp.*")

# Only used for the reliability flag (structural similarity to the training
# set), never for the model's features. Settings are irrelevant to the model,
# so they are fixed here.
_SIM_RADIUS, _SIM_BITS = 2, 2048
_sim_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=_SIM_RADIUS, fpSize=_SIM_BITS
)

_bundle = None     # cached model bundle
_train_fps = None  # cached training fingerprints ([] = unavailable)


# ============================================================
# LOADING
# ============================================================
def load_bundle(path=COX2_MODEL_FILE):
    """
    Returns the dict saved by cox_train.save_model():
        model         trained RandomForestClassifier
        feature_cols  exact column names and order the model expects
        cutoff        threshold for the 0/1 call
        split_mode    "random" or "scaffold"
        metrics       test-set scores
    The column list travels WITH the model, so featurize() can never silently
    hand it the wrong columns.
    """
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def load_model(path=COX2_MODEL_FILE):
    """Returns just the classifier (what shap.TreeExplainer wants)."""
    return load_bundle(path)["model"]


def model_info():
    """Prints how the loaded model was built and how well it scored."""
    b = load_bundle()
    print(f"features : {len(b['feature_cols'])} descriptor columns")
    print(f"split    : {b.get('split_mode', 'unknown')}")
    print(f"cutoff   : {b['cutoff']:.2f}")
    for k, v in b.get("metrics", {}).items():
        print(f"{k:12s}: {v:.3f}")
    print(f"reliability flag: "
          f"{'available' if _train_smiles_available() else 'NOT available'}")


def reset_cache():
    """Call after retraining, otherwise the old model stays in memory."""
    global _bundle, _train_fps
    _bundle, _train_fps = None, None


# ============================================================
# FEATURES
# ============================================================
def featurize(smiles_list):
    """
    SMILES -> numpy array (n_molecules x n_features), in exactly the column
    order the model was trained on.

    Selecting by feature_cols fails loudly with a KeyError if the saved model
    and the current cox_preprocess disagree — far better than silently
    feeding the model the wrong columns.

    Unparsable SMILES become all-zero rows, so row order is preserved. Use
    predict(), which marks those as NaN instead of scoring them.
    """
    b = load_bundle()
    df = compute_rdkit_features(smiles_list)
    missing = [c for c in b["feature_cols"] if c not in df.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} trained columns cannot be computed from SMILES: "
            f"{missing[:10]}. The model was trained on features this code "
            f"cannot reproduce — retrain, or add their computation to "
            f"cox_preprocess.compute_rdkit_features()."
        )
    return df[b["feature_cols"]].to_numpy(dtype=np.float32)


def feature_names():
    """
    Column names for SHAP plots, same length and order as featurize().

    These are RDKit descriptor names, so each one means exactly one thing:
    fr_phenol is a count of phenol groups, MolWt is molecular weight. No
    hashing caveat applies, unlike fingerprint bits.
    """
    return list(load_bundle()["feature_cols"])


# ============================================================
# RELIABILITY (applicability domain)
# ============================================================
def _train_smiles_available():
    """True if the training SMILES file exists and holds a smiles column."""
    try:
        return "smiles" in pd.read_csv(COX2_TRAIN_FILE, nrows=1).columns
    except Exception:
        return False


def _load_train_fps():
    """
    Fingerprints of the training compounds, for the similarity check.
    Returns [] when the training SMILES are unavailable.
    """
    global _train_fps
    if _train_fps is None:
        try:
            smis = pd.read_csv(COX2_TRAIN_FILE)["smiles"]
            _train_fps = [
                _sim_gen.GetFingerprint(m)
                for m in (Chem.MolFromSmiles(str(s)) for s in smis)
                if m is not None
            ]
        except Exception as exc:
            print(f"Reliability flag unavailable ({exc.__class__.__name__}): "
                  f"no training SMILES at {COX2_TRAIN_FILE}")
            _train_fps = []
    return _train_fps


# ============================================================
# PREDICTION
# ============================================================
def predict(smiles_list, cutoff=None, with_reliability=True):
    """
    Returns a DataFrame: smiles, proba, active, and where the training SMILES
    are available also max_sim_train and in_domain.

    proba is NaN for unparsable SMILES.
    max_sim_train = highest Tanimoto similarity to any training compound.

    Why the similarity column: the model scored AUC ~0.87 on test compounds
    similar to the training set, but only ~0.65 on dissimilar ones. Plant
    metabolites are often dissimilar, so filter to in_domain == True before
    drawing conclusions.

    Also note: the probabilities are a RANKING, not a verdict. On this data
    the model performs on par with a nearest-neighbour similarity baseline,
    so use the scores to compare and rank compounds, not to call one active.
    """
    smiles_list = list(smiles_list)
    bundle = load_bundle()
    cutoff = bundle["cutoff"] if cutoff is None else cutoff

    mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    valid = np.array([m is not None for m in mols])

    proba = np.full(len(smiles_list), np.nan)
    if valid.any():
        X = featurize([s for s, v in zip(smiles_list, valid) if v])
        proba[valid] = bundle["model"].predict_proba(X)[:, 1]

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
        if ref:
            out["max_sim_train"] = [
                np.nan
                if m is None
                else max(DataStructs.BulkTanimotoSimilarity(
                    _sim_gen.GetFingerprint(m), ref))
                for m in mols
            ]
            out["in_domain"] = out["max_sim_train"] >= SIM_CUTOFF

    return out


def predict_table(df, smiles_col="smiles", prefix="cox"):
    """
    Convenience for the plant data: scores a DataFrame and returns it with
    the prediction columns appended, keeping every original column.
    """
    preds = predict(df[smiles_col]).drop(columns="smiles")
    preds.columns = [f"{prefix}_{c}" for c in preds.columns]
    return pd.concat([df.reset_index(drop=True), preds], axis=1)
