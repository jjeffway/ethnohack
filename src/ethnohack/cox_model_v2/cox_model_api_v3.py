"""
cox_model_api.py
Loading and using the trained COX-2 classifier. No training code, no feature
definitions here — features come from cox_preprocess so that prediction can
never disagree with how the model was trained.

Pipeline: cox_preprocess.py -> cox_train.py -> this file.

Needs: rdkit, scikit-learn, joblib, numpy, pandas

Usage:
    import ethnohack.cox_model.cox_model_api as cox

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

    # Filter to in_domain == True first. Explaining predictions the model
    # cannot make reliably produces confident-looking but meaningless
    # rationales.
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

_bundle = None  # cached model bundle
_train_fps = None  # cached training fingerprints


# ============================================================
# LOADING
# ============================================================
def load_bundle(path=COX2_MODEL_FILE):
    """
    Returns the dict saved by cox_train.save_model():
        model         trained RandomForestClassifier
        feature_mode  "ecfp" | "ecfp_counts" | "ecfp_desc" | "ecfp_extended"
        feature_cols  exact column order the model expects
        fp_size, radius, cutoff, split_mode, metrics
    The feature settings travel WITH the model, so featurize() can never
    silently disagree with how the model was trained.
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
    print(f"feature mode : {b['feature_mode']} ({len(b['feature_cols'])} columns)")
    print(f"split        : {b['split_mode']}")
    print(f"cutoff       : {b['cutoff']:.2f}")
    for k, v in b["metrics"].items():
        print(f"{k:12s}: {v:.3f}")


def reset_cache():
    """Call after retraining, otherwise the old model stays in memory."""
    global _bundle, _train_fps
    _bundle, _train_fps = None, None


# ============================================================
# FEATURES  (thin wrappers — the real code lives in cox_preprocess)
# ============================================================
def _fp_generator(bundle):
    """Fingerprint generator matching the model's settings."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=bundle["radius"], fpSize=bundle["fp_size"]
    )


def fingerprint(smiles, bundle=None):
    """SMILES -> RDKit fingerprint object, or None if unparsable."""
    bundle = bundle or load_bundle()
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else _fp_generator(bundle).GetFingerprint(mol)


def featurize(smiles_list):
    """
    SMILES -> numpy array (n_molecules x n_features), in exactly the column
    order the model was trained on. Settings come from the saved bundle.
    Unparsable SMILES become all-zero rows, so row order is preserved.

    Selecting by feature_cols fails loudly with a KeyError if the saved model
    and the current cox_preprocess disagree — far better than silently
    feeding the model the wrong columns.
    """
    b = load_bundle()
    df = compute_rdkit_features(
        smiles_list, mode=b["feature_mode"], fp_size=b["fp_size"], radius=b["radius"]
    )
    return df[b["feature_cols"]].to_numpy(dtype=np.float32)


def feature_names():
    """
    Column names for SHAP plots, same length and order as featurize().

    Fingerprint columns are named fp_0 ... fp_2047. They are hashed, so one
    column can be set by different substructures. Treat a column as a hint
    about what the model used, not as proof of a specific fragment.
    """
    return list(load_bundle()["feature_cols"])


def explain_bit(bit_index, smiles_list, prefix="fp_"):
    """
    Optional helper for readable SHAP labels: finds a substructure that
    switches fingerprint column `bit_index` on, by scanning the given
    molecules. Returns a SMILES string, or None if no molecule sets that bit.
    """
    b = load_bundle()
    gen = _fp_generator(b)
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        info = rdFingerprintGenerator.AdditionalOutput()
        info.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=info)
        envs = info.GetBitInfoMap().get(bit_index)
        if not envs:
            continue
        atom, rad = envs[0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom) if rad else None
        return (
            Chem.MolFragmentToSmiles(mol, atomsToUse=[atom])
            if env is None
            else Chem.MolToSmiles(Chem.PathToSubmol(mol, env))
        )
    return None


def readable_names(smiles_list):
    """
    feature_names() with substructure hints added to the fingerprint columns,
    e.g. "fp_314 c1ccccc1". Slower than feature_names(); use it for the final
    SHAP figure rather than for every call.
    """
    names = feature_names()
    for i, n in enumerate(names):
        if n.startswith("fp_"):
            frag = explain_bit(int(n[3:]), smiles_list)
            if frag:
                names[i] = f"{n} {frag}"
    return names


# ============================================================
# PREDICTION
# ============================================================
def _load_train_fps(bundle):
    """Training fingerprints, for the applicability-domain check."""
    global _train_fps
    if _train_fps is None:
        smis = pd.read_csv(COX2_TRAIN_FILE)["smiles"]
        gen = _fp_generator(bundle)
        _train_fps = [
            gen.GetFingerprint(m)
            for m in (Chem.MolFromSmiles(str(s)) for s in smis)
            if m is not None
        ]
    return _train_fps


def predict(smiles_list, cutoff=None, with_reliability=True):
    """
    Returns a DataFrame: smiles, proba, active, max_sim_train, in_domain.

    proba is NaN for unparsable SMILES.
    max_sim_train = highest Tanimoto similarity to any training compound.

    Why the similarity column: the model scored AUC ~0.87 on test compounds
    similar to the training set, but only ~0.65 on dissimilar ones. Plant
    metabolites are often dissimilar, so filter to in_domain == True before
    drawing conclusions.

    Also note: the probabilities are a RANKING, not a verdict. The model
    performs on par with a nearest-neighbour similarity baseline on this data,
    so use the scores to compare and rank, not to call a compound active.
    """
    smiles_list = list(smiles_list)
    bundle = load_bundle()
    cutoff = bundle["cutoff"] if cutoff is None else cutoff

    gen = _fp_generator(bundle)
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
        ref = _load_train_fps(bundle)
        out["max_sim_train"] = [
            np.nan
            if m is None
            else max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), ref))
            for m in mols
        ]
        out["in_domain"] = out["max_sim_train"] >= SIM_CUTOFF

    return out
