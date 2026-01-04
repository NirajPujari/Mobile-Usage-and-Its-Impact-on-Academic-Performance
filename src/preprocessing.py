import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    title_mapping: dict[str, str],
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    df = df.rename(columns=title_mapping)

    drop_cols = {"Name", "Email", "Timestamp"}
    df = df.drop(columns=drop_cols & set(df.columns))

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(-1)
    return df
