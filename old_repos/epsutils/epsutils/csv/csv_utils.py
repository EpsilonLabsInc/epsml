import pandas as pd


def get_all_values(csv_file_name, column_name):
    df = pd.read_csv(csv_file_name, sep=",", low_memory=False)
    all_values = df[column_name].tolist()
    return all_values


def get_unique_values(csv_file_name, column_name):
    df = pd.read_csv(csv_file_name, sep=",", low_memory=False)
    unique_values = df[column_name].unique().tolist()
    return unique_values
