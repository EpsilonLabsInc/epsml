import argparse
import ast
from io import StringIO

import pandas as pd


def assign_projection_classifications(args):
    # Load input CSV file.
    print(f"Loading input CSV file {args.input_csv_path}")
    df = pd.read_csv(args.input_csv_path, low_memory=False)

    # Update image paths.
    print("Updating image paths")
    image_paths = df[args.image_paths_column_name].tolist()
    image_paths = [ast.literal_eval(image_path) if not pd.isna(image_path) else None for image_path in image_paths]

    # Load projection classifications CSV file(s).
    projection_classifications_dict = {}
    for i in range(len(args.projection_classifications_csv_paths)):
        print(f"Loading projection classifications CSV file {args.projection_classifications_csv_paths[i]}")
        with open(args.projection_classifications_csv_paths[i], "r") as f:
            filtered_lines = [line for line in f if ";" in line]
        projection_classifications_df = pd.read_csv(StringIO("".join(filtered_lines)), low_memory=False, header=None, sep=";")

        curr_dict = dict(zip(projection_classifications_df[0], projection_classifications_df[1]))
        overlap = set(projection_classifications_dict) & set(curr_dict)
        assert not overlap, f"Duplicate keys found in projection classifications: {overlap}"
        projection_classifications_dict = projection_classifications_dict | curr_dict

    # Assign projection classifications.
    projection_classifications = []
    print("Assigning projection classifications")
    for image_path in image_paths:
        if image_path is None:
            projection_classifications.append(None)
        else:
            values = [projection_classifications_dict[item] if item in projection_classifications_dict else None for item in image_path]
            projection_classifications.append(values)

    # Update projection classification column.
    print("Updating projection classification column")
    assert len(projection_classifications) == len(df)
    df["projection_classification"] = projection_classifications

    # Save output.
    print(f"Saving output file {args.output_csv_path}")
    df.to_csv(args.output_csv_path, index=False)


def main(args):
    assign_projection_classifications(args)


if __name__ == "__main__":
    INPUT_CSV_PATH = "/mnt/all-data/reports/gradient/gradient_cr_all_batches_tmp.csv"
    PROJECTION_CLASSIFICATIONS_CSV_PATHS = [
        "/mnt/all-data/reports/gradient/09JAN2025/projection_classificaton_results.csv",
        "/mnt/all-data/reports/gradient/20DEC2024/projection_classificaton_results.csv",
        "/mnt/all-data/reports/gradient/22JUL2024/projection_classificaton_results.csv"
    ]
    IMAGE_PATHS_COLUMN_NAME = "relative_image_paths"
    OUTPUT_CSV_PATH = "/mnt/all-data/reports/gradient/gradient_cr_all_batches_final.csv"

    args = argparse.Namespace(input_csv_path=INPUT_CSV_PATH,
                              projection_classifications_csv_paths=PROJECTION_CLASSIFICATIONS_CSV_PATHS,
                              image_paths_column_name=IMAGE_PATHS_COLUMN_NAME,
                              output_csv_path=OUTPUT_CSV_PATH)

    main(args)
