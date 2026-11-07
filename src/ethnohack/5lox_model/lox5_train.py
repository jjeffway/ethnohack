"""
lox5_train.py
Step 2 of 2: prepared table -> split -> tune -> evaluate -> save model.

Does NOT compute features. Run lox5_preprocess.py first.

Needs: scikit-learn, optuna, matplotlib, scipy, rdkit (for the similarity checks)

Usage (notebook):
    import ethnohack.cox_model.lox5_train as train

    data    = train.load_prepared()
    results = train.run_models(data, "both")   # random + scaffold, for the slide
    train.plot_roc(results)

    best = results[-1]                         # scaffold = the honest one
    sim  = train.similarity_baseline(best)     # is it better than similarity search?
    train.combined_baseline(best, sim)
    train.reliability_bins(best)               # WHERE does it work?
    train.save_model(best, tune_cutoff=True)
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from ethnohack.constants import (
    LOX5_CUTOFF,
    LOX5_MODEL_FILE,
    LOX5_PREPARED_CSV,
    LOX5_TRAIN_FILE,
    N_SPLITS,
    RANDOM_STATE,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
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
    cross_val_predict,
    cross_val_score,
)

from .lox5_preprocess import feature_columns

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fingerprints are used ONLY for the similarity checks below, never for the
# model's features, so these settings are fixed here and need no constant.
_SIM_RADIUS, _SIM_BITS = 2, 2048
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=_SIM_RADIUS, fpSize=_SIM_BITS
)


def load_prepared(path=LOX5_PREPARED_CSV):
    """Reads the table written by lox5_preprocess.build_dataset()."""
    data = pd.read_csv(path)
    print(f"Loaded {len(data)} compounds | {len(feature_columns(data))} features")
    return data


def _matrix(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else _fp_gen.GetFingerprint(mol)


def _has_smiles(result):
    """
    The similarity checks compare molecular structures, so they need SMILES.
    A descriptor-only table has none, and then these checks are skipped
    rather than crashing. Note the gap on the limitations slide.
    """
    return "smiles" in result["train_df"].columns


# ============================================================
# 1) SPLIT
# RANDOM:   close analogues land in BOTH sets -> the model recognises the
#           compound series instead of the chemistry -> flattering scores.
# SCAFFOLD: molecules sharing a ring core stay on one side -> closer to the
#           real task (unseen plant metabolites) -> lower but honest.
# ============================================================
def _make_cv(split_mode):
    cls = StratifiedGroupKFold if split_mode == "scaffold" else StratifiedKFold
    return cls(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def split_data(data, split_mode):
    splitter = _make_cv(split_mode)
    groups = data["scaffold"] if split_mode == "scaffold" else None
    tr_i, te_i = next(splitter.split(data, data["label"], groups=groups))
    return data.iloc[tr_i], data.iloc[te_i]


# ============================================================
# 2) TUNE AND FIT
# ============================================================
def suggest_params(trial):
    return {
        # more trees = more stable, just slower
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        # too deep memorises single compounds
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        # bigger leaves = smoother, less overfitting
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        # how many features each split may look at
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.1, 0.2]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        # "balanced" up-weights the smaller class instead of resampling
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def make_rf(params):
    # random_state fixed -> same data + same params = same forest, every time
    return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)


def tune(X_tr, y_tr, groups, split_mode, n_trials=30, timeout_min=15):
    """
    Hyperparameter search. The inner CV uses the SAME split logic as the outer
    split, otherwise tuning is optimistic even with a scaffold test set.

    Stops at n_trials OR timeout_min, whichever comes first. The timeout is
    checked between trials, so a running trial finishes (slight overrun).
    For exact reproducibility set timeout_min high and let n_trials decide.
    """
    cv = _make_cv(split_mode)
    cv_groups = groups if split_mode == "scaffold" else None

    def objective(trial):
        # average_precision = PR-AUC: the right metric when the actives matter
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


def evaluate(y_te, proba, cutoff=LOX5_CUTOFF):
    """Prints test metrics, returns them for saving."""
    pred = (proba >= cutoff).astype(int)
    m = {
        "prevalence": float(y_te.mean()),
        "pr_auc": average_precision_score(y_te, proba),
        "roc_auc": roc_auc_score(y_te, proba),
        "mcc": matthews_corrcoef(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }
    # Prevalence = share of actives. A coin flip already reaches this PR-AUC,
    # so compare your PR-AUC against this number, not against 0.
    print(f"\nPrevalence (random-guess PR-AUC): {m['prevalence']:.3f}")
    for k in ["pr_auc", "roc_auc", "mcc", "precision", "recall", "f1"]:
        print(f"{k:10s}: {m[k]:.3f}")
    return m


def run_split(data, split_mode, n_trials=30, timeout_min=15):
    """One full run = split -> tune -> fit -> evaluate."""
    print(f"\n{'=' * 55}\nSPLIT: {split_mode.upper()}\n{'=' * 55}")

    cols = feature_columns(data)
    train_df, test_df = split_data(data, split_mode)
    X_tr, y_tr = _matrix(train_df, cols), train_df["label"].values
    X_te, y_te = _matrix(test_df, cols), test_df["label"].values

    print(f"train: {len(train_df)} | test: {len(test_df)} | features: {len(cols)}")
    if split_mode == "scaffold":
        overlap = len(set(train_df["scaffold"]) & set(test_df["scaffold"]))
        print(f"scaffold overlap (must be 0): {overlap}")

    study = tune(
        X_tr, y_tr, train_df["scaffold"].values, split_mode, n_trials, timeout_min
    )

    model = make_rf(study.best_params)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, proba)

    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "split_mode": split_mode,
        "feature_cols": cols,
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "y_te": y_te,
        "proba": proba,
        "metrics": metrics,
        "auc": metrics["roc_auc"],
        "fpr": fpr,
        "tpr": tpr,
    }


def run_models(data, split_mode="both", n_trials=30, timeout_min=15):
    """
    Runs one or both split modes. The LAST result is the one to use.
    The gap between the two ROC curves IS the leakage a random split hides.
    Note: the limits apply per mode, so "both" takes up to twice as long.
    """
    modes = ["random", "scaffold"] if split_mode == "both" else [split_mode]
    return [run_split(data, m, n_trials, timeout_min) for m in modes]


# ============================================================
# 3) SANITY CHECKS
# ============================================================
def similarity_baseline(result):
    """
    Nearest-active-neighbour search: score each test compound by how similar
    it is to the closest known active. If the model barely beats this, it
    mostly learned "looks like a known active". Worth knowing and reporting.

    Returns None when the table has no SMILES column.
    """
    if not _has_smiles(result):
        print("Skipped: the similarity baseline needs a SMILES column.")
        return None

    fps_tr = [_fp(s) for s in result["train_df"]["smiles"]]
    fps_te = [_fp(s) for s in result["test_df"]["smiles"]]
    y_tr, y_te, proba = result["y_tr"], result["y_te"], result["proba"]

    actives = [fp for fp, y in zip(fps_tr, y_tr) if y == 1 and fp is not None]
    sim = np.array(
        [max(BulkTanimotoSimilarity(fp, actives)) if fp else 0.0 for fp in fps_te]
    )

    print(
        "Similarity baseline  ROC-AUC:",
        round(roc_auc_score(y_te, sim), 3),
        "| PR-AUC:",
        round(average_precision_score(y_te, sim), 3),
    )
    print(
        "Model                ROC-AUC:",
        round(roc_auc_score(y_te, proba), 3),
        "| PR-AUC:",
        round(average_precision_score(y_te, proba), 3),
    )
    return sim


def combined_baseline(result, sim):
    """
    Rank average of model and similarity. Caution: choosing this AFTER seeing
    the test scores is selection on the test set. Treat any gain as an
    observation, not a validated result.
    """
    if sim is None:
        print("Skipped: no similarity scores to combine with.")
        return None

    y_te = result["y_te"]
    combo = (rankdata(result["proba"]) + rankdata(sim)) / 2
    print(
        "Combined ROC-AUC:",
        round(roc_auc_score(y_te, combo), 3),
        "| PR-AUC:",
        round(average_precision_score(y_te, combo), 3),
    )
    return combo


def reliability_bins(result, n_bins=4):
    """
    Test AUC split by structural similarity to the training set.
    This is the key table: it shows WHERE the model works. The low-similarity
    bins predict how it will behave on unfamiliar plant metabolites.
    Uses fingerprints regardless of what the model trained on, because the
    question here is structural similarity, not model input.

    Returns None when the table has no SMILES column.
    """
    if not _has_smiles(result):
        print("Skipped: the reliability bins need a SMILES column.")
        return None

    fps_tr = [f for f in (_fp(s) for s in result["train_df"]["smiles"]) if f]
    fps_te = [_fp(s) for s in result["test_df"]["smiles"]]

    max_sim = np.array(
        [max(BulkTanimotoSimilarity(fp, fps_tr)) if fp else 0.0 for fp in fps_te]
    )

    df = pd.DataFrame({"y": result["y_te"], "p": result["proba"], "sim": max_sim})
    df["bin"] = pd.qcut(df["sim"], n_bins, duplicates="drop")

    rows = [
        (str(b), len(g), round(roc_auc_score(g["y"], g["p"]), 3))
        for b, g in df.groupby("bin", observed=True)
        if g["y"].nunique() == 2
    ]

    out = pd.DataFrame(rows, columns=["max Tanimoto to train", "n", "AUC"])
    print(out.to_string(index=False))
    return out


def plot_roc(results):
    plt.figure(figsize=(5, 4))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['split_mode']} (AUC {r['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


# ============================================================
# 4) SAVE
# ============================================================
def find_cutoff(result):
    """
    Picks the 0/1 threshold by cross-validation on the TRAINING data (MCC),
    so the test set stays untouched. 0.5 is rarely the best choice.
    """
    cv = _make_cv(result["split_mode"])
    groups = (
        result["train_df"]["scaffold"].values
        if result["split_mode"] == "scaffold"
        else None
    )
    oof = cross_val_predict(
        result["model"],
        result["X_tr"],
        result["y_tr"],
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    cuts = np.linspace(0.05, 0.95, 91)
    scores = [matthews_corrcoef(result["y_tr"], (oof >= c).astype(int)) for c in cuts]
    best = float(cuts[int(np.argmax(scores))])
    print(f"Tuned cutoff: {best:.2f} (CV MCC {max(scores):.3f})")
    return best


def save_model(result, tune_cutoff=False):
    """
    Saves the model together with everything needed to reproduce its input:
    the exact feature column order and the cutoff. Without this the API could
    hand the model different columns from the ones it was trained on.

    The training SMILES are written alongside when available; they are what
    the API's reliability flag compares new molecules against.
    """
    cutoff = find_cutoff(result) if tune_cutoff else LOX5_CUTOFF
    bundle = {
        "model": result["model"],
        "feature_cols": result["feature_cols"],  # exact column order
        "cutoff": cutoff,
        "split_mode": result["split_mode"],
        "metrics": result["metrics"],
    }
    LOX5_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, LOX5_MODEL_FILE)
    print(
        f"Saved: {LOX5_MODEL_FILE} (cutoff {cutoff:.2f}, "
        f"{len(bundle['feature_cols'])} features)"
    )

    if _has_smiles(result):
        LOX5_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        result["train_df"][["smiles"]].to_csv(LOX5_TRAIN_FILE, index=False)
        print(f"Saved: {LOX5_TRAIN_FILE} ({len(result['train_df'])} training SMILES)")
    else:
        print("No SMILES column: the reliability flag will be unavailable "
              "at prediction time.")
