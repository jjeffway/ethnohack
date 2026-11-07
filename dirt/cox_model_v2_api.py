"""
cox_model_api.py
Loading and using the trained COX-2 classifier. No training code here.

Needs: rdkit, scikit-learn, joblib, numpy, pandas

Usage:
    import ethnohack.cox_model.cox_model_api as cox

    df    = cox.predict(smiles_list)       # proba, active, reliability
    X     = cox.featurize(smiles_list)     # feature matrix for TreeSHAP
    names = cox.feature_names(smiles_list) # readable column names
    model = cox.load_model()               # sklearn RandomForestClassifier

TreeSHAP example:
    import shap
    bundle = cox.load_bundle()
    X      = cox.featurize(smiles_list)
    sv     = shap.TreeExplainer(bundle["model"]).shap_values(X)
    sv_act = sv[1] if isinstance(sv, list) else sv[..., 1]   # class "active"
    shap.summary_plot(sv_act, X, feature_names=cox.feature_names(smiles_list))
"""

import joblib
import numpy as np
import pandas as pd
from ethnohack.constants import COX2_MODEL_FILE, COX2_TRAIN_FILE, SIM_CUTOFF
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# 10 simple physicochemical properties, only used in mode "ecfp_desc"
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

_bundle = None  # cached model bundle
_train_fps = None  # cached training fingerprints


# ============================================================
# LOADING
# ============================================================
def load_bundle(path=COX2_MODEL_FILE):
    """
    Returns the saved dict:
        model         trained RandomForestClassifier
        feature_mode  "ecfp" | "ecfp_counts" | "ecfp_desc"
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


def reset_cache():
    """Call after retraining, otherwise the old model stays in memory."""
    global _bundle, _train_fps
    _bundle, _train_fps = None, None


# ============================================================
# FEATURES  (the single definition — training imports this one)
# ============================================================
def _fp_generator(bundle):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=bundle["radius"], fpSize=bundle["fp_size"]
    )


def fingerprint(smiles, radius=2, fp_size=2048):
    """SMILES -> RDKit fingerprint object, or None if unparsable."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return gen.GetFingerprint(mol)


def featurize(smiles_list, mode=None, fp_size=None, radius=None):
    """
    SMILES -> numpy array (n_molecules x n_features).
    Defaults come from the saved bundle, so inference always matches training.
    Training passes the values explicitly (no model exists yet).
    Unparsable SMILES become all-zero rows, so row order is preserved.
    """
    if mode is None or fp_size is None or radius is None:
        b = load_bundle()
        mode = b["feature_mode"] if mode is None else mode
        fp_size = b["fp_size"] if fp_size is None else fp_size
        radius = b["radius"] if radius is None else radius

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    n_desc = len(DESC) if mode == "ecfp_desc" else 0

    rows = []
    for s in smiles_list:
        vec = np.zeros((fp_size + n_desc,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            if mode == "ecfp_counts":
                # counts help when a substructure appearing 3x matters (polyphenols)
                vec[:fp_size] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr = np.zeros((fp_size,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
                vec[:fp_size] = arr
            if mode == "ecfp_desc":
                vec[fp_size:] = [f(mol) for _, f in DESC]
        rows.append(vec)
    return np.vstack(rows)


def feature_names(smiles_list, mode=None, fp_size=None, radius=None, prefix="bit"):
    """
    Readable column names for SHAP, same length and order as featurize().

    Each fingerprint bit that is switched on by one of the given molecules gets
    the SMILES of a substructure that sets it, e.g. "bit314 c1ccccc1".
    Caveat: bits are hashed, so different fragments can share a bit. Read the
    name as a hint about what the model used, not as proof.
    """
    if mode is None or fp_size is None or radius is None:
        b = load_bundle()
        mode = b["feature_mode"] if mode is None else mode
        fp_size = b["fp_size"] if fp_size is None else fp_size
        radius = b["radius"] if radius is None else radius

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    names = [f"{prefix}{i}" for i in range(fp_size)]
    seen = set()

    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        info = rdFingerprintGenerator.AdditionalOutput()
        info.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=info)
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

    if mode == "ecfp_desc":
        names += [n for n, _ in DESC]  # keeps names aligned with the extra columns
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
    drawing conclusions — explaining predictions the model cannot make
    reliably produces confident-looking but meaningless rationales.

    Also note: the probabilities are a RANKING, not a verdict. The model
    performs on par with a nearest-neighbour similarity baseline on this data.
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
