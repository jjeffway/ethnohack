"""
model.py
COX activity classifier (Random Forest on ECFP4, 2048 bits).

Needs: rdkit, scikit-learn, joblib, numpy, pandas, optuna, matplotlib, scipy
Paths and settings live in ethnohack.constants.

Usage (from a notebook):
    import ethnohack.cox_model.cox_model_api as cox
    data    = cox.add_scaffolds(cox.clean_data(cox.load_dataset()))
    results = cox.run_models(data, "scaffold")    # split -> tune -> fit -> evaluate
    cox.save_model(results[-1])
    X       = cox.featurize(smiles_list)          # feature matrix for TreeSHAP
    model   = cox.load_model()                    # sklearn RandomForestClassifier
    df      = cox.predict(smiles_list)            # proba, label, reliability
    names   = cox.bit_names(smiles_list)          # readable names for the 2048 features
"""

# load libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from ethnohack.constants import (
    COX2_CSV,
    COX2_CUTOFF,
    COX2_MODEL_FILE,
    COX2_TRAIN_FILE,
    FEATURE_MODE,
    FP_SIZE,
    MORGAN_RADIUS,
    N_SPLITS,
    RANDOM_STATE,
    SIM_CUTOFF,
)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

# silence RDKit and Optuna warnings
RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ECFP4 generator shared by all functions
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=FP_SIZE
)

# 10 simple physicochemical properties, only used if mode == "ecfp_desc"
DESC = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    rdMolDescriptors.CalcNumRings,
    rdMolDescriptors.CalcNumAromaticRings,
    rdMolDescriptors.CalcFractionCSP3,
    Descriptors.HeavyAtomCount,
]

# caches for the loaded model and the training fingerprints
_model = None
_train_fps = None


# ============================================================
# 1) DATA
#    SMILES are already canonical, so we only remove broken rows,
#    compounds with contradictory labels, and duplicates.
# ============================================================
def load_dataset():
    """Returns the cleaned COX2 dataset as a DataFrame."""
    return pd.read_csv(COX2_CSV)


def clean_data(df, smiles_col="Smiles", label_col="Activity"):
    """Returns a DataFrame with columns smiles, label (other columns are dropped)."""
    data = df[[smiles_col, label_col]].copy()
    data.columns = ["smiles", "label"]
    data["label"] = data["label"].astype(int)
    data = data.dropna(subset=["smiles"])

    ok = data["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    print("Unparsable SMILES removed:", int((~ok).sum()))
    data = data[ok]

    # same molecule labelled both 0 and 1 -> we cannot trust it, drop it
    conflict = data.groupby("smiles")["label"].nunique()
    data = data[~data["smiles"].isin(conflict[conflict > 1].index)]
    data = data.drop_duplicates(subset="smiles").reset_index(drop=True)
    print(f"Compounds: {len(data)} | actives: {data['label'].sum()}")
    print(f"({data['label'].mean():.1%})")
    return data


# Murcko scaffold = the ring core of a molecule (side chains stripped off)
def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    # ring-free molecules get their own group
    return scaf if scaf != "" else "ACYCLIC_" + smi


def add_scaffolds(data):
    """Adds a scaffold column (used to group molecules for scaffold splits)."""
    data = data.copy()
    data["scaffold"] = data["smiles"].apply(scaffold_of)
    return data


# ============================================================
# 2) FEATURES
#    A fingerprint turns a molecule into a fixed-length number vector.
#    ECFP4 = which small substructures are present around each atom.
# ============================================================
def _fp(smiles):
    """SMILES -> RDKit fingerprint object, or None if unparsable."""
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else _fp_gen.GetFingerprint(mol)


def featurize(smiles_list, mode=FEATURE_MODE):
    """
    SMILES -> numpy array (n_molecules x n_features).
    Random forests need no scaling, so we just stack columns.
    This is the X you pass to shap.TreeExplainer(model).shap_values(X).
    Unparsable SMILES become all-zero rows, so the row order is preserved.
    """
    n_desc = len(DESC) if mode == "ecfp_desc" else 0
    rows = []
    for s in smiles_list:
        vec = np.zeros((FP_SIZE + n_desc,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            if mode == "ecfp_counts":
                # counts can help when a substructure appearing 3x matters (e.g. polyphenols)
                vec[:FP_SIZE] = _fp_gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr = np.zeros((FP_SIZE,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(_fp_gen.GetFingerprint(mol), arr)
                vec[:FP_SIZE] = arr
            if mode == "ecfp_desc":
                # adds size/polarity info that fingerprints capture only indirectly
                vec[FP_SIZE:] = [f(mol) for f in DESC]
        rows.append(vec)
    return np.vstack(rows)


# ============================================================
# 3) TRAINING = split -> tune -> fit -> evaluate
# ============================================================
def _make_cv(split_mode):
    """Cross-validation splitter that matches the split mode."""
    if split_mode == "scaffold":
        return StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# Train/test split
# RANDOM: like train_test_split. Close analogues of the same compound land in
#         BOTH sets, so the model can recognise the series instead of the
#         chemistry -> scores look better than the model really is.
# SCAFFOLD: all molecules sharing a ring core stay on the same side. This is
#         closer to the real task (unseen plant metabolites) -> lower but honest.
def split_data(data, split_mode):
    splitter = _make_cv(split_mode)
    if split_mode == "scaffold":
        tr_i, te_i = next(splitter.split(data, data["label"], groups=data["scaffold"]))
    else:
        # 80/20, label-balanced
        tr_i, te_i = next(splitter.split(data, data["label"]))
    return data.iloc[tr_i], data.iloc[te_i]


def suggest_params(trial):
    """Hyperparameter search space for the Optuna trial."""
    return {
        # more trees = more stable, just slower; 300-900 is plenty here
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        # how deep a tree may grow; too deep memorises single compounds
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        # bigger leaves = smoother model, less overfitting
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        # how many of the 2048 bits each split may look at
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.1, 0.2]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        # "balanced" up-weights the smaller class instead of resampling the data
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def make_rf(params):
    """Builds the forest from a plain params dict."""
    return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)


def tune(X_tr, y_tr, groups, split_mode, n_trials=30, timeout_min=15):
    """Hyperparameter search. Returns the finished Optuna study."""
    # The inner cross-validation uses the SAME split logic as the outer split,
    # otherwise tuning would be optimistic even with a scaffold test set.
    cv = _make_cv(split_mode)
    cv_groups = groups if split_mode == "scaffold" else None

    def objective(trial):
        # average_precision = PR-AUC: the right metric when you care about the actives
        return cross_val_score(
            make_rf(suggest_params(trial)),
            X_tr,
            y_tr,
            groups=cv_groups,
            cv=cv,
            scoring="average_precision",
        ).mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout_min * 60, show_progress_bar=True
    )
    print("Best CV PR-AUC:", round(study.best_value, 3), "| params:", study.best_params)
    return study


def evaluate(y_te, proba, cutoff=COX2_CUTOFF):
    """Prints test metrics for predicted probabilities."""
    pred = (proba >= cutoff).astype(int)

    # Prevalence = share of actives. A coin flip would already reach this PR-AUC,
    # so always compare your PR-AUC against this number, not against 0.
    print(f"\nPrevalence (random-guess PR-AUC): {y_te.mean():.3f}")
    print("PR-AUC   :", round(average_precision_score(y_te, proba), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_te, proba), 3))
    print("MCC      :", round(matthews_corrcoef(y_te, pred), 3))
    print("Precision:", round(precision_score(y_te, pred, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_te, pred), 3))
    print("F1       :", round(f1_score(y_te, pred), 3))


def run_split(data, split_mode, n_trials=30, timeout_min=15, mode=FEATURE_MODE):
    """
    One full run = split -> tune -> fit -> evaluate.
    Returns a dict with the model, the data splits and the test predictions.
    """
    print(f"\n{'=' * 55}\nSPLIT: {split_mode.upper()}   FEATURES: {mode}\n{'=' * 55}")

    # ---- split ----
    train_df, test_df = split_data(data, split_mode)
    X_tr, y_tr = featurize(train_df["smiles"], mode), train_df["label"].values
    X_te, y_te = featurize(test_df["smiles"], mode), test_df["label"].values

    overlap = len(set(train_df["scaffold"]) & set(test_df["scaffold"]))
    print(
        f"train: {len(train_df)} | test: {len(test_df)} | scaffold overlap: {overlap}"
    )

    # ---- hyperparameter search ----
    study = tune(
        X_tr, y_tr, train_df["scaffold"].values, split_mode, n_trials, timeout_min
    )

    # ---- fit the best forest on all training data and test it ----
    model = make_rf(study.best_params)
    model.fit(X_tr, y_tr)

    # probability of being active
    proba = model.predict_proba(X_te)[:, 1]
    evaluate(y_te, proba)

    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "split_mode": split_mode,
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "y_te": y_te,
        "proba": proba,
        "auc": roc_auc_score(y_te, proba),
        "fpr": fpr,
        "tpr": tpr,
    }


def run_models(data, split_mode, n_trials=30, timeout_min=15):
    """
    Runs one or both split modes. Returns a list of results, the last one is
    the one to use (scaffold when split_mode == "both").
    """
    # Nice presentation slide: the gap between the two curves IS the leakage
    # a random split hides. Always report the scaffold number as your result.
    modes = ["random", "scaffold"] if split_mode == "both" else [split_mode]
    return [run_split(data, m, n_trials, timeout_min) for m in modes]


def plot_roc(results):
    """ROC curves for the results returned by run_models."""
    plt.figure(figsize=(5, 4))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['split_mode']} (AUC {r['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


def save_model(result):
    """Saves the model and its training SMILES (needed for the reliability flag)."""
    COX2_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], COX2_MODEL_FILE)
    result["train_df"][["smiles"]].to_csv(COX2_TRAIN_FILE, index=False)
    print(f"Saved: {COX2_MODEL_FILE}")


# ============================================================
# 4) BASELINE: nearest-active-neighbour similarity search
# ============================================================
def similarity_baseline(result):
    """Prints baseline vs model scores, returns the baseline score per test molecule."""
    fps_tr = [_fp(s) for s in result["train_df"]["smiles"]]
    fps_te = [_fp(s) for s in result["test_df"]["smiles"]]
    y_tr = result["train_df"]["label"].values
    y_te, proba = result["y_te"], result["proba"]

    fps_tr_active = [fp for fp, y in zip(fps_tr, y_tr) if y == 1]
    sim_to_active = np.array(
        [max(BulkTanimotoSimilarity(fp, fps_tr_active)) for fp in fps_te]
    )

    print("Similarity baseline ROC-AUC:", round(roc_auc_score(y_te, sim_to_active), 3))
    print(
        "Similarity baseline PR-AUC :",
        round(average_precision_score(y_te, sim_to_active), 3),
    )
    print("Your model      ROC-AUC:", round(roc_auc_score(y_te, proba), 3))
    print("Your model      PR-AUC :", round(average_precision_score(y_te, proba), 3))
    return sim_to_active


def combined_baseline(result, sim_to_active):
    """Prints scores for the rank average of model probability and similarity."""
    y_te = result["y_te"]
    combo = (rankdata(result["proba"]) + rankdata(sim_to_active)) / 2
    print("Combined ROC-AUC:", round(roc_auc_score(y_te, combo), 3))
    print("Combined PR-AUC :", round(average_precision_score(y_te, combo), 3))
    return combo


# ============================================================
# 5) USING THE SAVED MODEL
# ============================================================
def load_model():
    """Returns the trained RandomForestClassifier (cached)."""
    global _model
    if _model is None:
        _model = joblib.load(COX2_MODEL_FILE)
    return _model


def _load_train_fps():
    global _train_fps
    if _train_fps is None:
        smis = pd.read_csv(COX2_TRAIN_FILE)["smiles"]
        _train_fps = [f for f in (_fp(s) for s in smis) if f is not None]
    return _train_fps


def predict(smiles_list, cutoff=COX2_CUTOFF, with_reliability=True):
    """
    Returns a DataFrame with:
        smiles, proba, active (0/1), max_sim_train, in_domain
    proba is NaN for unparsable SMILES.
    max_sim_train = highest Tanimoto similarity to any training compound.
    The model is clearly weaker below ~0.5, so treat those predictions with care.
    Why the similarity? The model scored AUC ~0.87 on compounds similar to the
    training set but only ~0.65 on dissimilar ones. Plant metabolites are often
    dissimilar, so predictions with a low similarity are much less trustworthy.
    """
    smiles_list = list(smiles_list)
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
        out["in_domain"] = out["max_sim_train"] >= SIM_CUTOFF

    return out


def bit_names(smiles_list, prefix="bit"):
    """
    Readable names for the 2048 features, for SHAP plots.
    For every bit that is set in the given molecules, the name carries the SMILES
    of one substructure that switches that bit on, e.g. "bit314 c1ccccc1".
    Bits never set keep their plain name.
    """
    names = [f"{prefix}{i}" for i in range(FP_SIZE)]
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
