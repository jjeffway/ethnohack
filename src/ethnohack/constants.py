from pathlib import Path

# file directory
BASE_DIR = Path(__file__).resolve().parents[2]

# cox2
COX2_CSV = BASE_DIR / "datasets" / "cox2_cleaned.csv"
COX2_PREPARED_CSV = BASE_DIR / "datasets" / "cox2_prepared.csv"
COX2_MODEL_FILE = BASE_DIR / "src" / "ethnohack" / "cox_model" / "cox_model.joblib"
COX2_TRAIN_FILE = BASE_DIR / "src" / "ethnohack" / "cox_model" / "train_smiles_cox.csv"

# model settings
COX2_CUTOFF = (
    0.5  # probability cutoff for active, replace with the tuned cutoff if you have one
)
SIM_CUTOFF = 0.5  # min Tanimoto similarity to training set for in_domain
N_SPLITS = 5
RANDOM_STATE = 42


COCONUT_CSV = BASE_DIR / "datasets" / "coconut.csv"
COCONUT_SCORED_CSV = BASE_DIR / "datasets" / "coconut_scored.csv"

PLANT_GENERA_FILE = BASE_DIR / "datasets" / "plant_genera.txt"
