# EthnoHack Source Code

This directory contains the core Python source code for the EthnoHack project, implementing structure-based QSAR models for COX-2 and 5-LOX targets, with tools for scoring natural products from the COCONUT database.

## Directory Structure
```
src/ethnohack/
├── init.py              # Empty package initializer
├── constants.py             # Global configuration and path definitions
├── coconut.py               # COCONUT natural products loading and processing
├── explainer.py             # SHAP-based model interpretation and visualization
├── score_both.py            # Dual-target scoring for COCONUT compounds
│
├── cox_model/               # COX-2 (CHEMBL230) model
│   ├── init.py
│   ├── cox_preprocess.py    # Preprocessing for COX-2 training
│   ├── cox_train.py         # COX-2 model training with hyperparameter optimization
│   ├── cox_model_api.py     # COX-2 prediction API
│   └── cox_model.joblib     # Trained COX-2 Random Forest model
│
└── lox5_model/              # 5-LOX (CHEMBL215 / ALOX5) model
├── init.py
├── lox5_preprocess.py   # Preprocessing for 5-LOX training
├── lox5_train.py        # 5-LOX model training with hyperparameter optimization
├── lox5_model_api.py    # 5-LOX prediction API
└── 5lox_model.joblib    # Trained 5-LOX Random Forest model
```
## Core Modules

### constants.py

**Purpose**: Central configuration file containing all path definitions and global settings.

**Key Constants**:
- **Paths**: `BASE_DIR`, `COX2_CSV`, `LOX5_CSV`, `COX2_MODEL_FILE`, `LOX5_MODEL_FILE`, `COX2_TRAIN_FILE`, `LOX5_TRAIN_FILE`, `COCONUT_CSV`, `COCONUT_SCORED_CSV`, `PLANT_GENERA_FILE`
- **Model Settings**: `COX2_CUTOFF` (0.5), `LOX5_CUTOFF` (0.5), `SIM_CUTOFF` (0.5), `N_SPLITS` (5), `RANDOM_STATE` (42)

**Usage**:
```python
from ethnohack.constants import COX2_MODEL_FILE, RANDOM_STATE
```
## COX-2 model
### cox_preprocess.py

**Purpose**: Preprocess COX-2 data for training. Assumes descriptor columns already exist in input (from data_pipeline.py).


**Usage:**
```python
import ethnohack.cox_model.cox_preprocess as prep

# Full preprocessing
data = prep.build_dataset(path="data/processed/cox2_model_table.csv")

# Or step-by-step
raw = prep.load_raw()
cleaned = prep.clean_data(raw)
final = prep.add_scaffolds(cleaned)

# Verify features match
prep.verify_features(final)
```
### cox_train.py

**Purpose:** Train COX-2 Random Forest classifier with hyperparameter optimization using Optuna.

**Usage:**
```python
import ethnohack.cox_model.cox_train as train

# Load data
data = train.load_prepared()

# Train with scaffold split (recommended)
results = train.run_models(data, split_mode="scaffold", n_trials=30, timeout_min=15)

# Save the best model (last result is scaffold split)
best = results[-1]
train.save_model(best, tune_cutoff=True)

# Plot ROC
train.plot_roc(results)

# Evaluate baselines
sim = train.similarity_baseline(best)
train.combined_baseline(best, sim)
train.reliability_bins(best)
```

### cox_model_api.py

**Purpose:** Load and use trained COX-2 model for prediction. Provides the prediction interface for scoring new compounds.

**Usage:**

```python
import ethnohack.cox_model.cox_model_api as cox

# Get model info
cox.model_info()

# Predict on SMILES list
df = cox.predict(["CCO", "CCC"], with_reliability=True)
print(df[["smiles", "proba", "active", "max_sim_train", "in_domain"]])

# Get feature matrix for SHAP
X = cox.featurize(["CCO", "CCC"])
names = cox.feature_names()

# Load model directly
model = cox.load_model()
```

## 5-LOX Model (lox5_model/)

The lox5_model/ directory contains identical functionality to cox_model/ but for the 5-LOX (ALOX5 / CHEMBL215) target:

`lox5_preprocess.py`: Same as cox_preprocess.py <br>
`lox5_train.py`: Same as cox_train.py <br>
`lox5_model_api.py`: Same as cox_model_api.py <br>
`5lox_model.joblib`: Trained 5-LOX model <br>
`5lox_prepared.csv`: Prepared 5-LOX dataset



## COCONUT Processing
### coconut.py

**Purpose:** Load, clean, filter, and score COCONUT natural products database with the trained COX-2 model.

Pipeline:

1) Load COCONUT CSV
2) Standardize SMILES
3) Annotate with plant taxonomy
4) Collapse stereoisomers/salts
5) Filter to plant compounds
6) Remove training overlap
7) Score with COX-2 model
8) Add confidence tiers
9) Flag alerts (PAINS, Brenk, custom)

**Usage:**
```python
import ethnohack.coconut as coco

# Load plant genera
genera = coco.load_plant_genera("data/raw/plant_genera.txt")

# Prepare COCONUT
screening_table, overlap_table = coco.prepare_coconut(
    path="data/raw/coconut.csv", plant_genera=genera, organism_col="organisms"
)

# Score
scored = coco.score_coconut(screening_table)

# Add alerts (on candidates only)
candidates = scored[scored["max_sim_train"] >= 0.49]
candidates = coco.flag_alerts(candidates)

# Get top hits
hits = coco.top_hits(candidates, n=50, in_domain_only=True, exclude_alerts=True)

# Save
coco.save_scores(scored, path="data/external/coconut_scored.csv")
```

Confidence Tiers (based on max Tanimoto similarity to training):

    low: < 0.49
    medium: 0.49-0.61
    good: 0.61-0.72
    high: ≥ 0.72

## score_both.py
**Purpose:** Score COCONUT natural products with both COX-2 and 5-LOX models in a single pass, reusing computed descriptors and fingerprints for efficiency.

**Usage:**
```python
import ethnohack.score_both as sb

# Score COCONUT with both models
out = sb.run(
    coconut_path="data/raw/coconut.csv",
    out_path="data/external/coconut_scored_dual_scored.csv",
    limit=2000,  # Test run
    chunk_size=5000,
)

# Get summary
sb.summarize(out)

# Top dual hits
hits = sb.top_hits(out, n=50, dual_only=True, in_domain_only=True)
```

## explainer.py

**Purpose:** SHAP-based model interpretation and visualization for understanding model predictions.

**Supported Model Types:**

    Descriptor models: RDKit descriptor columns (MolWt, TPSA, fr_* fragment counts)
    Fingerprint models: Morgan fingerprint bits (fp_0, fp_1, ...)


**Usage:**
```python
import ethnohack.cox_model.cox_model_api as cox
import ethnohack.explainer as exp

# Load model
model = cox.load_model()
feature_names = cox.feature_names()

# Explain one molecule
exp.explain_molecule(
    smiles="CC(=O)c1ccccc1C(=O)O",  # Aspirin
    model=model,
    target_class=1,  # Active class
    feature_names=feature_names,
    n_top=5,
)

# Global SHAP for training data
import ethnohack.cox_model.cox_preprocess as prep

data = prep.build_dataset()
X = cox.featurize(data["smiles"])
shap_matrix, _ = exp.global_shap(model, X, target_class=1, n_max=500)
exp.plot_global_shap(shap_matrix, X, feature_names=feature_names, max_display=20)

# Top features
top = exp.top_features_table(shap_matrix, X, feature_names=feature_names, n_top=15)
print(top)
```
Color Coding:

    Red: Feature pushes prediction toward active
    Blue: Feature pushes prediction toward inactive
    Gray: Feature has no effect (SHAP value ≈ 0)

Note: Fragment highlighting requires RDKit's fragment SMARTS patterns file.