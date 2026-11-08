"""
coconut.py
Load natural products from a COCONUT CSV download, clean them up, keep only
plant compounds, remove anything the model was trained on, and score the rest
with the saved COX-2 model.

Download the CSV from https://coconut.naturalproducts.net/download and put it
at COCONUT_CSV. Use a download that includes the ORGANISM column (check with
list_columns()), otherwise the plant filter has nothing to work with.

Needs: pandas, numpy, rdkit, joblib (>=1.3), tqdm, and the saved model from
cox_train.save_model()

Usage (notebook):
    import ethnohack.coconut as coco

    # one-off: where are the organism names?
    coco.list_columns()

    # a set of plant genus names, e.g. from a GBIF / WFO export
    genera = coco.load_plant_genera("plant_genera.txt")

    # load -> standardise -> plants only -> collapse stereoisomers
    #      -> remove compounds the model was trained on
    coconut, overlap = coco.prepare_coconut(plant_genera=genera)

    scored = coco.score_coconut(coconut)        # also adds confidence tiers
    coco.summarize(scored)

    # optional, slow-ish: PAINS / reactivity / size flags on the candidates
    cand = scored[scored["max_sim_train"] >= 0.49].reset_index(drop=True)
    cand = coco.flag_alerts(cand)

    hits = coco.top_hits(cand, n=50, in_domain_only=False,
                         min_confidence="medium", exclude_alerts=True)
    plant_table = coco.plant_compound_table(cand)   # input for pair analysis
    coco.save_scores(scored)
"""

import csv
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ethnohack.cox_model.cox_model_api as cox
import numpy as np
import pandas as pd
from ethnohack.constants import COCONUT_CSV, COCONUT_SCORED_CSV, RANDOM_STATE
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

# RDKit prints a warning for every odd molecule, which floods the notebook
RDLogger.DisableLog("rdApp.*")

# max-Tanimoto-to-training edges for the confidence tiers. These are the bin
# edges from the reliability table of the scaffold-split run (0.49 / 0.61 /
# 0.72). Re-derive them if the model is retrained.
CONFIDENCE_EDGES = (0.49, 0.61, 0.72)
CONFIDENCE_LABELS = ["low", "medium", "good", "high"]

# a compound this similar to a training molecule is a near-copy, not a prediction
NEAR_DUPLICATE_SIM = 0.95

# substructures that make an "active" call suspect (reactive or promiscuous)
CUSTOM_ALERTS = {
    "methylene_lactone": "[CH2]=[C;R]C(=O)[O;R]",  # alpha-methylene lactone
    "catechol": "c([OH])c([OH])",
    "galloyl": "[OH]c1cc(C(=O)[#8,#7])cc([OH])c1[OH]",
    "quinone": "[#6]1(=O)[#6]=,:[#6][#6](=O)[#6]=,:[#6]1",
    "o_quinone": "[#6]1(=O)[#6](=O)[#6]=,:[#6][#6]=,:[#6]1",
    "nitroaromatic": "c[N+](=O)[O-]",
}

_LARGEST = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------


# print the columns of the COCONUT CSV, to find the SMILES / organism columns
def list_columns(path=COCONUT_CSV):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    print(cols)
    guesses = [c for c in cols if "organism" in c.lower() or "taxon" in c.lower()]
    print("Possible organism columns:", guesses or "none found")
    return cols


# read the COCONUT CSV -> DataFrame with id, name, smiles (and organisms)
def load_coconut(
    path=COCONUT_CSV,
    smiles_col="canonical_smiles",
    id_col="identifier",
    name_col="name",
    organism_col=None,
    n_max=None,
):
    # check the column names first, so a wrong name gives a helpful error
    available = pd.read_csv(path, nrows=0).columns.tolist()
    if smiles_col not in available:
        raise KeyError(f"'{smiles_col}' not in CSV. Columns: {available}")
    if organism_col is not None and organism_col not in available:
        raise KeyError(f"'{organism_col}' not in CSV. Columns: {available}")

    # read only the columns we need, the lite CSV has many more
    wanted = {smiles_col, id_col, name_col}
    if organism_col is not None:
        wanted.add(organism_col)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df = df.rename(columns={smiles_col: "smiles", id_col: "id", name_col: "name"})

    # id and name are optional
    for col in ("id", "name"):
        if col not in df.columns:
            df[col] = None

    # drop missing and duplicate structures (exact string match only, the
    # stereo-aware collapse happens after standardisation)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset="smiles")

    # optional random subsample, useful for a quick test run
    if n_max is not None and len(df) > n_max:
        df = df.sample(n=n_max, random_state=RANDOM_STATE)

    cols = ["id", "name", "smiles"]
    if organism_col is not None:
        cols.append(organism_col)
    df = df[cols].reset_index(drop=True)
    print(f"Loaded {len(df)} natural products")
    return df


# ---------------------------------------------------------------------------
# 2. Standardisation
# ---------------------------------------------------------------------------


# one SMILES -> parent structure as canonical SMILES, or None if unusable.
# Steps: RDKit cleanup, keep the largest fragment (drops salts and solvents),
# neutralise charges, optionally strip stereochemistry. Tautomers are NOT
# canonicalised (slow, and ECFP bits rarely care).
def standardize_smiles(smiles, strip_stereo=True):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        if strip_stereo:
            Chem.RemoveStereochemistry(mol)
        out = Chem.MolToSmiles(mol, isomericSmiles=not strip_stereo)
    except Exception:
        return None
    return out or None


def _standardize_chunk(smiles_list, strip_stereo):
    return [standardize_smiles(s, strip_stereo) for s in smiles_list]


# add std_smiles (and parent_id) to a DataFrame, drop unusable structures.
# Runs in parallel: n_jobs=-1 uses every core.
def standardize_df(
    df, smiles_col="smiles", n_jobs=-1, chunk_size=5000, strip_stereo=True
):
    smiles = df[smiles_col].tolist()
    chunks = [smiles[i : i + chunk_size] for i in range(0, len(smiles), chunk_size)]
    jobs = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_standardize_chunk)(c, strip_stereo) for c in chunks
    )
    std = []
    for part in tqdm(jobs, total=len(chunks), desc="Standardizing"):
        std.extend(part)

    out = df.copy()
    out["std_smiles"] = std

    # COCONUT ids look like CNP0188888.2: the part before the dot is the
    # parent compound, the number after it is a stereo variant
    if "id" in out.columns:
        out["parent_id"] = out["id"].astype("string").str.split(".").str[0]

    n_bad = int(out["std_smiles"].isna().sum())
    out = out.dropna(subset=["std_smiles"]).reset_index(drop=True)
    print(f"Standardized: {len(out)} kept | {n_bad} unusable removed")
    return out


# merge rows that share a std_smiles (stereoisomers, salt forms, duplicates).
# The plant lists are united, so a compound keeps every plant it was reported
# in. all_ids lists the merged identifiers, n_variants how many were merged.
def collapse_duplicates(df, key="std_smiles"):
    dup = df.duplicated(key, keep=False)
    singles = df[~dup].copy()
    singles["n_variants"] = 1
    singles["all_ids"] = singles["id"].astype("string")

    multi = df[dup]
    if multi.empty:
        print(f"Collapsed duplicates: {len(df)} -> {len(singles)}")
        return singles.reset_index(drop=True)

    def _union(series):
        seen, items = set(), []
        for value in series.dropna():
            for p in str(value).split("|"):
                if p and p not in seen:
                    seen.add(p)
                    items.append(p)
        return "|".join(items)

    # "first" skips missing values, so a name from any variant is kept
    agg = {c: "first" for c in df.columns if c != key}
    if "plants" in df.columns:
        agg["plants"] = _union
    grouped = multi.groupby(key, sort=False)
    merged = grouped.agg(agg)
    merged["n_variants"] = grouped.size()
    merged["all_ids"] = grouped["id"].agg(lambda s: "|".join(s.astype(str)))
    merged = merged.reset_index()

    out = pd.concat([singles, merged[singles.columns]], ignore_index=True)
    if "n_plant_species" in out.columns:
        out["n_plant_species"] = [
            len(p.split("|")) if isinstance(p, str) and p else 0 for p in out["plants"]
        ]
        out["is_plant"] = out["n_plant_species"] > 0
    print(f"Collapsed duplicates: {len(df)} -> {len(out)}")
    return out


# ---------------------------------------------------------------------------
# 3. Plant filter
# ---------------------------------------------------------------------------


# a set of plant genus names from a file. Two layouts:
#   .txt            one genus per line
#   .csv / .tsv     a table; give genus_col, and kingdom_col to keep only
#                   plants (e.g. the GBIF backbone Taxon.tsv has columns
#                   "genus" and "kingdom" with value "Plantae")
def load_plant_genera(
    path,
    genus_col="genus",
    kingdom_col=None,
    kingdom_values=("Plantae", "Viridiplantae"),
    sep=None,
):
    path = Path(path)
    if kingdom_col is None and path.suffix.lower() == ".txt":
        genera = {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}
    else:
        tab = path.suffix.lower() in (".tsv", ".tab")
        if sep is None:
            sep = "\t" if tab else ","
        usecols = [genus_col] + ([kingdom_col] if kingdom_col else [])
        kwargs = {"quoting": csv.QUOTE_NONE} if tab else {}
        genera = set()
        for chunk in pd.read_csv(
            path, sep=sep, usecols=usecols, chunksize=500_000, dtype=str, **kwargs
        ):
            if kingdom_col:
                chunk = chunk[chunk[kingdom_col].isin(kingdom_values)]
            names = chunk[genus_col].dropna().str.strip().unique()
            genera.update(n for n in names if n and " " not in n)
    print(f"Plant genera loaded: {len(genera)}")
    return genera


_TAXON_RE = re.compile(
    r"^\s*(?:[x\u00d7]\s+)?([A-Z][A-Za-z\-]+)"  # genus (optional hybrid sign)
    r"(?:\s+(?:[x\u00d7]\s+)?([a-z][a-z\-]+))?"  # species epithet
)
_NOT_EPITHETS = {"sp", "spp", "cf", "aff", "var", "subsp"}


# "Mentha x piperita L." -> ("Mentha", "Mentha piperita")
def _parse_taxon(text):
    m = _TAXON_RE.match(text or "")
    if m is None:
        return None
    genus, epithet = m.group(1), m.group(2)
    if epithet and epithet not in _NOT_EPITHETS:
        return genus, f"{genus} {epithet}"
    return genus, genus


# build the plant genus set WITHOUT a big taxonomy download: count the genera
# in the COCONUT organism column, look the most frequent ones up in the GBIF
# species-match web service (needs internet, a few minutes), keep those whose
# kingdom is Plantae, and save them one per line to out_file.
#   top_n       how many of the most frequent genera to look up. The printed
#               coverage shows what share of all organism entries they cover.
#   GBIF "Plantae" includes land plants and green/red algae, not brown algae
#   or fungi. Genus names that exist in several kingdoms get GBIF's single
#   best match, so skim the result for homonyms.
def plant_genera_from_gbif(
    path=COCONUT_CSV,
    organism_col="organisms",
    sep=r"[|;]",
    top_n=5000,
    workers=8,
    out_file=None,
    kingdoms=("Plantae",),
):
    orgs = pd.read_csv(path, usecols=[organism_col], low_memory=False)[organism_col]
    counts = Counter()
    for value in orgs.dropna():
        for token in re.split(sep, value):
            parsed = _parse_taxon(token)
            if parsed is not None:
                counts[parsed[0]] += 1

    top = counts.most_common(top_n)
    total = sum(counts.values())
    covered = sum(c for _, c in top)
    print(
        f"{len(counts)} distinct genera | looking up the top {len(top)} "
        f"(covers {covered / max(total, 1):.1%} of organism entries)"
    )

    def _lookup(genus):
        url = "https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode(
            {"name": genus}
        )
        for _ in range(2):  # one retry
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
                # accept only a genus-level exact hit
                if data.get("rank") == "GENUS" and data.get("canonicalName") == genus:
                    return genus, data.get("kingdom"), None
                return genus, None, None
            except Exception as exc:  # network hiccup, rate limit, ...
                err = str(exc)
        return genus, None, err

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(_lookup, [g for g, _ in top]), total=len(top)):
            results.append(r)

    failed = [g for g, _, err in results if err]
    plants = sorted(g for g, k, _ in results if k in kingdoms)
    kingdom_counts = Counter(k for _, k, _ in results)
    print("Kingdom of looked-up genera:", dict(kingdom_counts.most_common(8)))
    print(f"Plant genera: {len(plants)} | request failures: {len(failed)}")
    if failed:
        print("Re-run to retry the failures, e.g.:", failed[:5])

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(plants) + "\n")
        print(f"Saved: {out_file}")
    return set(plants)


# add the plant species each compound was reported in.
# Adds: plants ("|"-joined species names), n_plant_species, is_plant.
# A species counts as a plant when its GENUS is in plant_genera. Genus names
# shared with animals or fungi (homonyms) can slip through; list known
# offenders in exclude_genera. Check the printed "not matched" genera to see
# what the plant list is missing.
def annotate_plants(
    df, plant_genera, organism_col="organisms", sep=r"[|;]", exclude_genera=()
):
    if organism_col not in df.columns:
        raise KeyError(
            f"'{organism_col}' not in the table. Reload with "
            "load_coconut(organism_col=...); list_columns() shows the options."
        )
    genera = set(plant_genera) - set(exclude_genera)

    plants, unmatched = [], Counter()
    for value in df[organism_col]:
        found, seen = [], set()
        if isinstance(value, str):
            for token in re.split(sep, value):
                parsed = _parse_taxon(token)
                if parsed is None:
                    continue
                genus, name = parsed
                if genus in genera:
                    if name not in seen:
                        seen.add(name)
                        found.append(name)
                else:
                    unmatched[genus] += 1
        plants.append("|".join(found))

    out = df.copy()
    out["plants"] = plants
    out["n_plant_species"] = [len(p.split("|")) if p else 0 for p in plants]
    out["is_plant"] = out["n_plant_species"] > 0

    n_org = int(out[organism_col].notna().sum())
    print(
        f"With organism annotation: {n_org} ({n_org / max(len(out), 1):.1%}) | "
        f"with a plant species: {int(out['is_plant'].sum())}"
    )
    print("Most frequent genera NOT matched as plants:")
    print(", ".join(f"{g} ({c})" for g, c in unmatched.most_common(15)))
    return out


# one row per (plant, compound). Input for the per-plant pair / dual-target
# analysis. A compound found in 3 plants appears 3 times.
def plant_compound_table(df, cols=None):
    if "plants" not in df.columns:
        raise KeyError("Run annotate_plants() first.")
    if cols is None:
        cols = [
            "id",
            "name",
            "std_smiles",
            "proba",
            "rank_pct",
            "max_sim_train",
            "confidence",
            "has_alert",
        ]
    t = df.assign(plant=df["plants"].str.split("|")).explode("plant")
    t = t[t["plant"].notna() & (t["plant"] != "")]
    keep = ["plant"] + [c for c in cols if c in t.columns]
    return t[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Remove compounds the model was trained on
# ---------------------------------------------------------------------------


# the raw training table (smiles + label), read through the preprocessing module
def load_training_reference(smiles_col="Smiles", label_col="Activity"):
    import ethnohack.cox_model.cox_preprocess as prep

    raw = prep.load_raw()
    return raw[[smiles_col, label_col]].rename(
        columns={smiles_col: "smiles", label_col: "label"}
    )


# split the table into
#   kept    compounds the model has NOT seen (use these for screening)
#   overlap compounds whose standardised structure is in the training data,
#           with train_label (mean label if the training set lists it more
#           than once) and train_n. These are a labelled natural-product set:
#           useful as an external test if you retrain without them.
# train_df needs columns "smiles" and "label".
def split_training_overlap(
    df, train_df=None, smiles_col="std_smiles", n_jobs=-1, strip_stereo=True
):
    if train_df is None:
        train_df = load_training_reference()
    ref = standardize_df(train_df, n_jobs=n_jobs, strip_stereo=strip_stereo)
    ref_keys = ref.groupby("std_smiles").agg(
        train_label=("label", "mean"), train_n=("label", "size")
    )

    in_train = df[smiles_col].isin(ref_keys.index)
    overlap = df[in_train].join(ref_keys, on=smiles_col)
    kept = df[~in_train]
    print(
        f"Training overlap: {len(overlap)} removed from screening | {len(kept)} remain"
    )
    return kept.reset_index(drop=True), overlap.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Full preparation in one call
# ---------------------------------------------------------------------------


# load -> standardise -> annotate plants -> collapse stereoisomers
#      -> keep plants -> remove training compounds.
# Returns (screening_table, training_overlap_table).
# Collapsing happens BEFORE the plant filter, so a stereoisomer without an
# organism tag still inherits the plants of its siblings.
def prepare_coconut(
    path=COCONUT_CSV,
    plant_genera=None,
    plants_only=True,
    organism_col="organisms",
    exclude_genera=(),
    train_df=None,
    strip_stereo=True,
    n_jobs=-1,
    n_max=None,
    **load_kwargs,
):
    if plants_only and plant_genera is None:
        raise ValueError(
            "plants_only=True needs plant_genera (see load_plant_genera), "
            "or pass plants_only=False to skip the plant filter."
        )

    df = load_coconut(
        path,
        organism_col=organism_col if plants_only else None,
        n_max=n_max,
        **load_kwargs,
    )
    df = standardize_df(df, n_jobs=n_jobs, strip_stereo=strip_stereo)

    if plants_only:
        df = annotate_plants(
            df, plant_genera, organism_col=organism_col, exclude_genera=exclude_genera
        )
    df = collapse_duplicates(df)

    if plants_only:
        df = df[df["is_plant"]].reset_index(drop=True)
        print(f"Plant compounds: {len(df)}")

    return split_training_overlap(
        df, train_df=train_df, n_jobs=n_jobs, strip_stereo=strip_stereo
    )


# ---------------------------------------------------------------------------
# 6. Scoring
# ---------------------------------------------------------------------------


# score all molecules in chunks (keeps memory flat for large files).
# Scores the standardised structure when present, so the model sees the same
# kind of SMILES as the training data went through and duplicates stay merged.
def score_coconut(df, chunk_size=5000, with_reliability=True, smiles_col=None):
    if smiles_col is None:
        smiles_col = "std_smiles" if "std_smiles" in df.columns else "smiles"

    parts = []
    for start in tqdm(range(0, len(df), chunk_size), desc="Scoring"):
        chunk = df.iloc[start : start + chunk_size]

        # predict: proba, active, and (optionally) max_sim_train, in_domain
        pred = cox.predict(chunk[smiles_col], with_reliability=with_reliability)

        # keep the original row labels so the join below lines up
        pred.index = chunk.index
        parts.append(pred.drop(columns="smiles"))

    scored = df.join(pd.concat(parts))
    if with_reliability:
        scored = add_confidence(scored)
    return scored


# add confidence tiers, a near-duplicate flag and a rank percentile.
#   confidence       low / medium / good / high, by max similarity to training
#   near_duplicate   True if max_sim_train >= near_dup (a copy, not a prediction)
#   rank_pct         percentile of proba among scored molecules (1.0 = best).
#                    Compare these, not raw probabilities: the forest outputs
#                    are not calibrated, so multiplying two models' probas is
#                    not a joint probability.
def add_confidence(
    scored, edges=CONFIDENCE_EDGES, near_dup=NEAR_DUPLICATE_SIM, proba_col="proba"
):
    out = scored.copy()
    sim = out["max_sim_train"]
    out["confidence"] = pd.cut(
        sim, bins=[-np.inf, *edges, np.inf], labels=CONFIDENCE_LABELS
    )
    out["near_duplicate"] = sim >= near_dup
    out["rank_pct"] = out[proba_col].rank(pct=True)
    return out


# ---------------------------------------------------------------------------
# 7. Alerts (reactive / promiscuous / oversized)
# ---------------------------------------------------------------------------

_CATALOGS = {}


def _catalogs():
    if not _CATALOGS:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        for name in ("PAINS", "BRENK"):
            params = FilterCatalogParams()
            params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
            _CATALOGS[name] = FilterCatalog(params)
    return _CATALOGS["PAINS"], _CATALOGS["BRENK"]


# flag compounds whose predicted activity may be assay interference.
# Adds: pains, brenk, custom_alerts (each "|"-joined), mw, has_alert.
#   pains         PAINS filter matches
#   brenk         Brenk unwanted-substructure matches (a softer, longer list)
#   custom_alerts CUSTOM_ALERTS: methylene lactones, catechols, galloyl esters,
#                 quinones, nitroaromatics
#   has_alert     PAINS or custom alert (Brenk alone does not set it)
# Single process, roughly a millisecond per molecule: run it on the candidates
# (e.g. max_sim_train >= 0.49), not on all of COCONUT.
def flag_alerts(df, smiles_col="std_smiles"):
    pains_cat, brenk_cat = _catalogs()
    patterns = {k: Chem.MolFromSmarts(v) for k, v in CUSTOM_ALERTS.items()}

    rows = []
    for smi in tqdm(df[smiles_col], desc="Alerts"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(("", "", "", np.nan))
            continue
        pains = sorted({m.GetDescription() for m in pains_cat.GetMatches(mol)})
        brenk = sorted({m.GetDescription() for m in brenk_cat.GetMatches(mol)})
        custom = [k for k, q in patterns.items() if mol.HasSubstructMatch(q)]
        rows.append(
            ("|".join(pains), "|".join(brenk), "|".join(custom), Descriptors.MolWt(mol))
        )

    out = df.copy()
    flags = pd.DataFrame(
        rows, columns=["pains", "brenk", "custom_alerts", "mw"], index=out.index
    )
    out = out.join(flags)
    out["has_alert"] = (out["pains"] != "") | (out["custom_alerts"] != "")
    return out


# ---------------------------------------------------------------------------
# 8. Summary, ranking, saving
# ---------------------------------------------------------------------------


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

    if "confidence" in scored.columns:
        print("\nBy confidence tier (max similarity to training):")
        counts = scored["confidence"].value_counts().reindex(CONFIDENCE_LABELS)
        act = (
            scored[scored["active"] == 1]["confidence"]
            .value_counts()
            .reindex(CONFIDENCE_LABELS)
        )
        print(pd.DataFrame({"molecules": counts, "predicted_active": act}).fillna(0))
    if "near_duplicate" in scored.columns:
        print(
            f"Near-duplicates of training (sim >= {NEAR_DUPLICATE_SIM}): "
            f"{int(scored['near_duplicate'].sum())}"
        )


# highest-scoring molecules (the probability is a ranking, not a verdict)
#   in_domain_only         the model's own in_domain flag (default on, as before)
#   min_confidence         "low" / "medium" / "good" / "high": keep that tier and
#                          above. Combine with in_domain_only=False for a softer
#                          cut than the model's in_domain flag.
#   exclude_near_duplicates drop copies of training molecules
#   exclude_alerts         drop compounds with a PAINS / custom alert
#                          (needs flag_alerts first)
#   max_mw                 drop very large molecules (needs flag_alerts first)
def top_hits(
    scored,
    n=50,
    in_domain_only=True,
    min_confidence=None,
    exclude_near_duplicates=True,
    exclude_alerts=False,
    max_mw=None,
):
    hits = scored.dropna(subset=["proba"])

    # predictions far from the training data are much less reliable
    if in_domain_only:
        hits = hits[hits["in_domain"]]

    if min_confidence is not None:
        allowed = CONFIDENCE_LABELS[CONFIDENCE_LABELS.index(min_confidence) :]
        hits = hits[hits["confidence"].isin(allowed)]

    if exclude_near_duplicates and "near_duplicate" in hits.columns:
        hits = hits[~hits["near_duplicate"]]

    if exclude_alerts:
        if "has_alert" not in hits.columns:
            raise KeyError("Run flag_alerts() before exclude_alerts=True.")
        hits = hits[~hits["has_alert"]]

    if max_mw is not None:
        if "mw" not in hits.columns:
            raise KeyError("Run flag_alerts() before max_mw=...")
        hits = hits[hits["mw"] <= max_mw]

    return hits.sort_values("proba", ascending=False).head(n).reset_index(drop=True)


# write all scores to disk
def save_scores(scored, path=COCONUT_SCORED_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    print(f"Saved: {path}")
