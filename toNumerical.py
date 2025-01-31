import pandas as pd
from util import load_json, load_csv


def preprocess_data(
    df: pd.DataFrame,
    output_file: str,
    title_mapping: dict[str, str],
    mappings: dict[str, dict[str, int]],
) -> None:
    """Preprocess the dataset by renaming columns, mapping values, and handling missing values."""

    df = df.rename(columns=title_mapping)
    df.drop(columns=["Name", "Email", "Timestamp"], inplace=True)

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Handle missing values by filling them with a placeholder, e.g., -1
    df = df.fillna(-1)
    df.to_csv(output_file, index=False)
    print(
        f"Preprocessing complete. The numerical data has been saved to: {output_file}"
    )


if __name__ == "__main__":
    title_mapping: dict[str, str] = load_json(
        "C:/Users/pujar/Desktop/Project/DS/json/title_mapping.json"
    )
    mappings: dict[str, dict[str, int]] = load_json(
        "C:/Users/pujar/Desktop/Project/DS/json/mappings.json"
    )
    file: dict[str, str] = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    preprocess_data(load_csv(file["input"]), file["output"], title_mapping, mappings)
