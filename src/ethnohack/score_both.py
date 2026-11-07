"""
score_both.py
Scores COCONUT natural products with BOTH trained models (COX-2 and 5-LOX)
and writes one row per compound.

Why one script instead of running each notebook's section 9 twice: the
expensive steps are the RDKit descriptors and the Morgan fingerprint, and both
models need the same ones. Computing them once and reusing them for both
targets roughly halves the runtime.

Run from a notebook:
    import ethnohack.score_both as sb
    out = sb.run(limit=2000)          # test on a sample first
    out = sb.run()                    # then the full file

Or from a terminal:
    python -m ethnohack.score_both --limit 2000
    python -m ethnohack.score_both

Output columns
    id, name, parent_id            compound identity
    std_smiles                     structure actually scored
    organisms, plants, ...         carried through for the plant analysis
    cox2_proba, lox5_proba         probability of activity (a RANKING, not a verdict)
    cox2_active, lox5_active       0/1 at each model's tuned cutoff
    cox2_max_sim, lox5_max_sim     highest Tanimoto to that model's training set
    cox2_in_domain, lox5_in_domain whether the prediction is in a reliable range
    dual_active                    flagged active on BOTH targets
    dual_in_domain                 reliable on BOTH targets
    dual_score                     min(cox2_proba, lox5_proba) -- a dual
                                   inhibitor is only as good as its weaker side
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from ethnohack.constants import (
    COX2_MODEL_FILE,
    COX2_TRAIN_FILE,
    LOX5_MODEL_FILE,
    LOX5_TRAIN_FILE,
    SIM_CUTOFF,
)
from ethnohack.cox_model.cox_preprocess import compute_rdkit_features

RDLogger.DisableLog("rdApp.*")

# Fingerprints are used ONLY for the similarity / applicability-domain check,
# never as model features, so the settings are fixed here.
_sim_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Score columns from an earlier single-target run. They are dropped on load,
# because after this script "proba" would be ambiguous between two targets.
STALE_COLS = [
    "proba", "active", "max_sim_train", "in_domain",
    "confidence", "near_duplicate", "rank_pct",
]

# Metadata worth keeping for the plant-level analysis that comes next.
KEEP_COLS = [
    "id", "name", "parent_id", "std_smiles", "smiles",
    "organisms", "plants", "n_plant_species", "is_plant",
    "n_variants", "all_ids",
]


# ============================================================
# TARGETS
# ============================================================
class Target:
    """One trained model plus the training fingerprints it is compared against."""

    def __init__(self, name, model_file, train_file):
        self.name = name
        bundle = joblib.load(model_file)
        self.model = bundle["model"]
        self.cols = list(bundle["feature_cols"])
        self.cutoff = float(bundle["cutoff"])
        self.metrics = bundle.get("metrics", {})
        self.train_fps = self._load_train_fps(train_file)

        print(f"{name}: {len(self.cols)} features | cutoff {self.cutoff:.2f} "
              f"| {len(self.train_fps)} training fingerprints")
        if self.metrics:
            print(f"  test ROC-AUC {self.metrics.get('roc_auc', float('nan')):.3f} "
                  f"| PR-AUC {self.metrics.get('pr_auc', float('nan')):.3f}")

    @staticmethod
    def _load_train_fps(train_file):
        """Training fingerprints for the applicability-domain check."""
        try:
            smis = pd.read_csv(train_file)["smiles"]
        except Exception as exc:
            print(f"  WARNING: no training SMILES at {train_file} "
                  f"({exc.__class__.__name__}): similarity will be NaN")
            return []
        return [
            _sim_gen.GetFingerprint(m)
            for m in (Chem.MolFromSmiles(str(s)) for s in smis)
            if m is not None
        ]

    def predict(self, desc_df, valid_mask):
        """
        desc_df: the full RDKit descriptor table for this chunk.
        Selecting this model's own columns is what keeps training and
        prediction aligned; a KeyError here means the saved model expects
        something compute_rdkit_features() no longer produces.
        """
        proba = np.full(len(desc_df), np.nan)
        if valid_mask.any():
            X = desc_df.loc[valid_mask, self.cols].to_numpy(dtype=np.float32)
            proba[valid_mask.to_numpy()] = self.model.predict_proba(X)[:, 1]
        return proba

    def similarity(self, fps):
        """Highest Tanimoto similarity of each molecule to this target's training set."""
        if not self.train_fps:
            return np.full(len(fps), np.nan)
        return np.array([
            np.nan if fp is None
            else max(DataStructs.BulkTanimotoSimilarity(fp, self.train_fps))
            for fp in fps
        ])


# ============================================================
# SCORING
# ============================================================
def load_coconut(path, limit=None, seed=42):
    """Loads the COCONUT table, drops stale single-target score columns."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    dropped = [c for c in STALE_COLS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"Dropped stale score columns from an earlier run: {dropped}")

    if "std_smiles" not in df.columns:
        raise KeyError("No 'std_smiles' column; that is the structure to score.")

    # one row per compound: parent_id already collapses variants (salts,
    # stereoisomers) if the COCONUT prep did that, otherwise fall back
    key = "parent_id" if "parent_id" in df.columns else "std_smiles"
    before = len(df)
    df = df.drop_duplicates(subset=key).reset_index(drop=True)
    if len(df) < before:
        print(f"Deduplicated on {key!r}: {before} -> {len(df)} compounds")

    if limit and limit < len(df):
        df = df.sample(limit, random_state=seed).reset_index(drop=True)
        print(f"Sampled {limit} compounds for a test run")

    return df


def score_chunk(chunk, targets):
    """
    Scores one chunk on every target.
    Descriptors and fingerprints are computed ONCE here and reused, which is
    the whole point of scoring both targets in one pass.
    """
    smiles = chunk["std_smiles"].astype(str).tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = pd.Series([m is not None for m in mols])

    desc = compute_rdkit_features(smiles)            # once, shared by both models
    fps = [None if m is None else _sim_gen.GetFingerprint(m) for m in mols]

    out = {}
    for t in targets:
        proba = t.predict(desc, valid)
        sim = t.similarity(fps)
        out[f"{t.name}_proba"] = proba
        out[f"{t.name}_active"] = np.where(
            np.isnan(proba), np.nan, (proba >= t.cutoff).astype(float)
        )
        out[f"{t.name}_max_sim"] = sim
        out[f"{t.name}_in_domain"] = sim >= SIM_CUTOFF

    return pd.DataFrame(out, index=chunk.index)


def add_dual_columns(df, targets):
    """Combined flags across both targets."""
    names = [t.name for t in targets]
    if len(names) < 2:
        return df

    a, b = names[0], names[1]
    df["dual_active"] = (df[f"{a}_active"] == 1) & (df[f"{b}_active"] == 1)
    df["dual_in_domain"] = df[f"{a}_in_domain"] & df[f"{b}_in_domain"]

    # A compound is only a dual inhibitor to the extent of its WEAKER target,
    # so the minimum is the right summary, not the mean.
    df["dual_score"] = df[[f"{a}_proba", f"{b}_proba"]].min(axis=1)

    # percentile rank, so the number is readable without knowing the scale
    df["dual_rank_pct"] = df["dual_score"].rank(pct=True)
    return df


def run(coconut_path, out_path=None, limit=None, chunk_size=5000, save_every=20):
    """
    Full run: load -> score both targets in chunks -> flags -> save.
    Partial results are written every `save_every` chunks, so a crash or an
    interrupted kernel does not lose the work done so far.
    """
    coconut_path = Path(coconut_path)
    out_path = Path(out_path) if out_path else coconut_path.with_name(
        coconut_path.stem + "_dual_scored.csv"
    )

    print("=" * 60)
    targets = [
        Target("cox2", COX2_MODEL_FILE, COX2_TRAIN_FILE),
        Target("lox5", LOX5_MODEL_FILE, LOX5_TRAIN_FILE),
    ]
    print("=" * 60)

    df = load_coconut(coconut_path, limit=limit)
    keep = [c for c in KEEP_COLS if c in df.columns]

    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    print(f"\nScoring {len(df)} compounds in {len(chunks)} chunks of {chunk_size}")

    start = time.time()
    scored = []
    for i, chunk in enumerate(chunks, 1):
        scored.append(score_chunk(chunk, targets))

        done = sum(len(s) for s in scored)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0
        remaining = (len(df) - done) / rate / 60 if rate else 0
        print(f"  chunk {i}/{len(chunks)} | {done}/{len(df)} | "
              f"{rate:.0f} mol/s | ~{remaining:.0f} min left", flush=True)

        if i % save_every == 0 or i == len(chunks):
            partial = pd.concat(
                [df[keep].iloc[:done], pd.concat(scored)], axis=1
            )
            add_dual_columns(partial, targets).to_csv(out_path, index=False)

    out = add_dual_columns(
        pd.concat([df[keep], pd.concat(scored)], axis=1), targets
    )
    out = out.sort_values("dual_score", ascending=False).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    summarize(out)
    return out


# ============================================================
# SUMMARY
# ============================================================
def summarize(out):
    """
    Prints what the scores look like. The in-domain fractions are the numbers
    that matter most: they say how much of this natural-product space the
    models can actually speak to.
    """
    n = len(out)
    print(f"\n{'=' * 60}\nScored {n} compounds\n{'=' * 60}")

    for t in ("cox2", "lox5"):
        if f"{t}_proba" not in out.columns:
            continue
        p, dom = out[f"{t}_proba"], out[f"{t}_in_domain"]
        print(f"{t}: mean proba {p.mean():.3f} | "
              f"active {int((out[f'{t}_active'] == 1).sum())} "
              f"({(out[f'{t}_active'] == 1).mean():.1%}) | "
              f"in domain {dom.mean():.1%} | "
              f"median similarity {out[f'{t}_max_sim'].median():.2f}")

    if "dual_active" in out.columns:
        print(f"\ndual active        : {int(out['dual_active'].sum())} "
              f"({out['dual_active'].mean():.2%})")
        print(f"dual in domain     : {int(out['dual_in_domain'].sum())} "
              f"({out['dual_in_domain'].mean():.2%})")
        both = out["dual_active"] & out["dual_in_domain"]
        print(f"dual active AND reliable: {int(both.sum())}")

    print("\nReminder: these probabilities are a RANKING, not a verdict. Both "
          "models perform close to a nearest-neighbour similarity baseline, so "
          "treat hits as hypotheses and compare plants rather than trusting "
          "single compounds.")

    low = [t for t in ("cox2", "lox5")
           if f"{t}_in_domain" in out.columns and out[f"{t}_in_domain"].mean() < 0.3]
    if low:
        print(f"\nNOTE: most compounds are OUT of domain for {low}. Those "
              f"predictions sit in the weaker performance range, so say so "
              f"when you report the results.")


def top_hits(out, n=50, dual_only=True, in_domain_only=False):
    """The highest-ranked compounds, for a quick look or a slide."""
    sel = out
    if dual_only and "dual_active" in sel.columns:
        sel = sel[sel["dual_active"]]
    if in_domain_only and "dual_in_domain" in sel.columns:
        sel = sel[sel["dual_in_domain"]]

    cols = [c for c in (
        "id", "name", "plants", "n_plant_species",
        "cox2_proba", "lox5_proba", "dual_score",
        "cox2_max_sim", "lox5_max_sim", "dual_in_domain",
    ) if c in sel.columns]

    return sel.sort_values("dual_score", ascending=False).head(n)[cols]


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Score COCONUT natural products with both target models."
    )
    parser.add_argument("coconut", nargs="?", help="path to the COCONUT CSV")
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument("--limit", type=int, default=None,
                        help="score a random sample of this many compounds")
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()

    path = args.coconut
    if path is None:
        from ethnohack.constants import COCONUT_SCORES_FILE
        path = COCONUT_SCORES_FILE

    run(path, out_path=args.out, limit=args.limit, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
