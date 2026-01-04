from src.util import load_csv, load_json
from src.preprocessing import preprocess_data
from src.eda import perform_eda
from src.clustering import perform_clustering
from src.prediction import perform_prediction
from src.config import DATASET_DIR, JSON_DIR, INPUT_DIR, OUTPUT_DIR


def main():
    paths = {"input": INPUT_DIR, "output": OUTPUT_DIR}
    raw_df = load_csv(DATASET_DIR / paths["input"])

    title_mapping = load_json(JSON_DIR / "title_mapping.json")
    mappings = load_json(JSON_DIR / "mappings.json")

    df = preprocess_data(raw_df, title_mapping, mappings)

    perform_eda(df)
    df = perform_clustering(df)
    perform_prediction(df)


if __name__ == "__main__":
    main()
