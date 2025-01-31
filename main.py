from toNumerical import preprocess_data
from util import load_json, load_csv
from eda import perform_eda
from clustering import perform_clustering
from predication import perform_prediction


def main():
    # Load file paths from JSON
    file_paths = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    input_csv = load_csv(file_paths["input"])

    # Step 0: Perform Preprocessing
    print("\nStarting Preprocessing...")
    title_mapping: dict[str, str] = load_json(
        "C:/Users/pujar/Desktop/Project/DS/json/title_mapping.json"
    )
    mappings: dict[str, dict[str, int]] = load_json(
        "C:/Users/pujar/Desktop/Project/DS/json/mappings.json"
    )
    preprocess_data(input_csv, file_paths["output"], title_mapping, mappings)
    df = load_csv(file_paths["output"])

    # Step 1: Perform Exploratory Data Analysis (EDA)
    # print("\nStarting EDA...")
    # perform_eda(df)

    # Step 2: Perform Clustering
    print("\nStarting Clustering...")
    df = perform_clustering(df)

    # Step 3: Perform Prediction (Machine Learning)
    print("\nStarting Prediction...")
    perform_prediction(df)


if __name__ == "__main__":
    main()
