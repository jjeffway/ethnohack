"""
coconut_screen.py
Load natural products from a COCONUT CSV download and score them with the
saved COX-2 model.

Download the CSV (lite format is enough) from
https://coconut.naturalproducts.net/download and put it at COCONUT_CSV.

Needs: pandas, tqdm, and the saved model from cox_train.save_model()

Usage (notebook):
    import ethnohack.coconut_screen as coco

    coconut = coco.load_coconut()
    scored  = coco.score_coconut(coconut)
    coco.summarize(scored)
    hits    = coco.top_hits(scored, n=50)
    coco.save_scores(scored)
"""

import ethnohack.cox_model.cox_model_api as cox
import pandas as pd
from ethnohack.constants import COCONUT_CSV, COCONUT_SCORED_CSV, RANDOM_STATE
from tqdm.auto import tqdm


# read the COCONUT CSV -> DataFrame with id, name, smiles
def load_coconut(
    path=COCONUT_CSV,
    smiles_col="canonical_smiles",
    id_col="identifier",
    name_col="name",
    n_max=None,
):
    # check the column names first, so a wrong name gives a helpful error
    available = pd.read_csv(path, nrows=0).columns.tolist()
    if smiles_col not in available:
        raise KeyError(f"'{smiles_col}' not in CSV. Columns: {available}")

    # read only the columns we need, the lite CSV has many more
    wanted = {smiles_col, id_col, name_col}
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df = df.rename(columns={smiles_col: "smiles", id_col: "id", name_col: "name"})

    # id and name are optional
    for col in ("id", "name"):
        if col not in df.columns:
            df[col] = None

    # drop missing and duplicate structures
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset="smiles")

    # optional random subsample, useful for a quick test run
    if n_max is not None and len(df) > n_max:
        df = df.sample(n=n_max, random_state=RANDOM_STATE)

    df = df[["id", "name", "smiles"]].reset_index(drop=True)
    print(f"Loaded {len(df)} natural products")
    return df


# score all molecules in chunks (keeps memory flat for large files)
def score_coconut(df, chunk_size=5000, with_reliability=True):
    parts = []
    for start in tqdm(range(0, len(df), chunk_size), desc="Scoring"):
        chunk = df.iloc[start : start + chunk_size]

        # predict: proba, active, and (optionally) max_sim_train, in_domain
        pred = cox.predict(chunk["smiles"], with_reliability=with_reliability)

        # keep the original row labels so the join below lines up
        pred.index = chunk.index
        parts.append(pred.drop(columns="smiles"))

    return df.join(pd.concat(parts))


# print how many molecules were scored, were in domain, and were predicted active
def summarize(scored):
    n_total = len(scored)
    n_scored = int(scored["proba"].notna().sum())
    print(
        f"Molecules: {n_total} | scored: {n_scored} | unparsable: {n_total - n_scored}"
    )

    # in_domain = similar enough to the training data for the score to be trusted
    if "in_domain" in scored.columns:
        n_dom = int(scored["in_domain"].sum())
        print(f"In domain: {n_dom} ({n_dom / max(n_scored, 1):.1%})")
        n_act_dom = int(((scored["active"] == 1) & scored["in_domain"]).sum())
        print(f"Predicted active and in domain: {n_act_dom}")

    print(f"Predicted active (all): {int((scored['active'] == 1).sum())}")


# highest-scoring molecules (the probability is a ranking, not a verdict)
def top_hits(scored, n=50, in_domain_only=True):
    hits = scored.dropna(subset=["proba"])

    # predictions far from the training data are much less reliable
    if in_domain_only:
        hits = hits[hits["in_domain"]]

    return hits.sort_values("proba", ascending=False).head(n).reset_index(drop=True)


# write all scores to disk
def save_scores(scored, path=COCONUT_SCORED_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    print(f"Saved: {path}")
