import pandas as pd


def get_all_values(csv_file_name, column_name):
    df = pd.read_csv(csv_file_name, sep=",", low_memory=False)
    all_values = df[column_name].tolist()
    return all_values


def get_unique_values(csv_file_name, column_name):
    df = pd.read_csv(csv_file_name, sep=",", low_memory=False)
    unique_values = df[column_name].unique().tolist()
    return unique_values


def subtract_matching_rows(original_df: pd.DataFrame, rows_to_remove: pd.DataFrame) -> pd.DataFrame:
    merged = original_df.merge(rows_to_remove, how="outer", indicator=True)
    filtered_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return filtered_df
