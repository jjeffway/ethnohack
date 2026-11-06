"""
cox_train.py
Trains the COX-2 classifier: clean -> split -> tune -> fit -> evaluate -> save.

Needs everything cox_model_api needs, plus: optuna, matplotlib, scipy

Usage (from a notebook):
    import ethnohack.cox_model.cox_train as train

    data    = train.add_scaffolds(train.clean_data(train.load_dataset()))
    results = train.run_models(data, "both")      # random + scaffold, for the slide
    train.plot_roc(results)

    best = results[-1]                            # scaffold = the honest one
    sim  = train.similarity_baseline(best)        # is the model better than similarity?
    train.combined_baseline(best, sim)
    train.reliability_bins(best)                  # where the model actually works
    train.save_model(best, tune_cutoff=True)
"""

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
)
from rdkit import Chem
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
    cross_val_predict,
    cross_val_score,
)

# one shared definition of featurize / fingerprint — never duplicate it
from .. import cox_model_v2_api as api

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _featurize(smiles_list, mode=FEATURE_MODE):
    """Training-time wrapper: passes settings explicitly (no saved model yet)."""
    return api.featurize(smiles_list, mode=mode, fp_size=FP_SIZE, radius=MORGAN_RADIUS)


def _fp(smiles):
    return api.fingerprint(smiles, radius=MORGAN_RADIUS, fp_size=FP_SIZE)


# ============================================================
# 1) DATA
# ============================================================
def load_dataset():
    return pd.read_csv(COX2_CSV)


def clean_data(df, smiles_col="Smiles", label_col="Activity"):
    """Returns a DataFrame with columns smiles, label. SMILES are already canonical."""
    data = df[[smiles_col, label_col]].copy()
    data.columns = ["smiles", "label"]
    data["label"] = data["label"].astype(int)
    data = data.dropna(subset=["smiles"])

    ok = data["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    print("Unparsable SMILES removed:", int((~ok).sum()))
    data = data[ok]

    # same molecule labelled both 0 and 1 -> untrustworthy, drop it
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
    data = data.copy()
    data["scaffold"] = data["smiles"].apply(scaffold_of)
    return data


# ============================================================
# 2) SPLIT
# RANDOM:   close analogues land in BOTH sets -> the model recognises the
#           series instead of the chemistry -> scores look better than reality.
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
# 3) TUNE AND FIT
# ============================================================
def suggest_params(trial):
    return {
        # more trees = more stable, just slower
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        # too deep memorises single compounds
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        # bigger leaves = smoother, less overfitting
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
    return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)


def tune(X_tr, y_tr, groups, split_mode, n_trials=30, timeout_min=15):
    """
    Hyperparameter search. The inner CV uses the SAME split logic as the outer
    split, otherwise tuning is optimistic even with a scaffold test set.
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


def evaluate(y_te, proba, cutoff=COX2_CUTOFF):
    """Prints test metrics. Returns them as a dict for saving."""
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


def run_split(data, split_mode, n_trials=30, timeout_min=15, mode=FEATURE_MODE):
    """One full run = split -> tune -> fit -> evaluate."""
    print(f"\n{'=' * 55}\nSPLIT: {split_mode.upper()}   FEATURES: {mode}\n{'=' * 55}")

    train_df, test_df = split_data(data, split_mode)
    X_tr, y_tr = _featurize(train_df["smiles"], mode), train_df["label"].values
    X_te, y_te = _featurize(test_df["smiles"], mode), test_df["label"].values

    print(f"train: {len(train_df)} | test: {len(test_df)} | features: {X_tr.shape[1]}")
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
        "feature_mode": mode,
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


def run_models(data, split_mode, n_trials=30, timeout_min=15):
    """
    Runs one or both split modes. The last result is the one to use.
    Running "both" gives the presentation slide: the gap between the curves
    IS the leakage a random split hides. Report the scaffold number.
    """
    modes = ["random", "scaffold"] if split_mode == "both" else [split_mode]
    return [run_split(data, m, n_trials, timeout_min) for m in modes]


# ============================================================
# 4) SANITY CHECKS
# ============================================================
def similarity_baseline(result):
    """
    Nearest-active-neighbour search: score each test compound by its similarity
    to the closest known active. If the model barely beats this, it mostly
    learned "looks like a known active" — worth knowing and worth reporting.
    """
    fps_tr = [_fp(s) for s in result["train_df"]["smiles"]]
    fps_te = [_fp(s) for s in result["test_df"]["smiles"]]
    y_tr, y_te, proba = result["y_tr"], result["y_te"], result["proba"]

    actives = [fp for fp, y in zip(fps_tr, y_tr) if y == 1 and fp is not None]
    sim = np.array(
        [
            max(BulkTanimotoSimilarity(fp, actives)) if fp is not None else 0.0
            for fp in fps_te
        ]
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
    Rank average of model and similarity. Note: picking this AFTER seeing the
    test scores is selection on the test set — treat any gain as an
    observation, not a validated result.
    """
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
    Test AUC split by similarity to the training set. Shows where the model
    actually works — the key table for judging plant predictions.
    """
    fps_tr = [_fp(s) for s in result["train_df"]["smiles"]]
    fps_te = [_fp(s) for s in result["test_df"]["smiles"]]
    fps_tr = [f for f in fps_tr if f is not None]

    max_sim = np.array(
        [
            max(BulkTanimotoSimilarity(fp, fps_tr)) if fp is not None else 0.0
            for fp in fps_te
        ]
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
# 5) SAVE
# ============================================================
def find_cutoff(result):
    """
    Picks the 0/1 threshold by cross-validation on the TRAINING data, using MCC.
    Done on training data so the test set stays untouched.
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
    Saves model + the settings needed to reproduce its features, so the API
    can never featurize differently from how the model was trained.
    Also writes the training SMILES, needed for the reliability flag.
    """
    cutoff = find_cutoff(result) if tune_cutoff else COX2_CUTOFF
    bundle = {
        "model": result["model"],
        "feature_mode": result["feature_mode"],
        "fp_size": FP_SIZE,
        "radius": MORGAN_RADIUS,
        "cutoff": cutoff,
        "split_mode": result["split_mode"],
        "metrics": result["metrics"],
    }
    COX2_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, COX2_MODEL_FILE)
    result["train_df"][["smiles"]].to_csv(COX2_TRAIN_FILE, index=False)
    api.reset_cache()  # otherwise the old model stays cached in memory
    print(
        f"Saved: {COX2_MODEL_FILE}  (mode={bundle['feature_mode']}, "
        f"cutoff={cutoff:.2f})"
    )
