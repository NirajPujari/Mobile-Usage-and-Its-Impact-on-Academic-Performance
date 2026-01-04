import json
import pandas as pd


def load_json(file_path: str):
    """Load JSON data from a file."""
    with open(file_path, "r") as file:
        return json.load(file)


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)
