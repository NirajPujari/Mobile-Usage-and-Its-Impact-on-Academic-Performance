from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = BASE_DIR / "dataset" / "survey_data.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "numerical_data.csv"
JSON_DIR = BASE_DIR / "json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_CLUSTERS = 10
RIDGE_ALPHA = 10
