import argparse
import ast
from io import StringIO

import pandas as pd


def assign_chest_classifications(args):
    # Load input CSV file.
    print(f"Loading input CSV file {args.input_csv_path}")
    df = pd.read_csv(args.input_csv_path, low_memory=False)

    # Update image paths.
    print("Updating image paths")
    image_paths = df[args.image_paths_column_name].tolist()
    image_paths = [ast.literal_eval(image_path) for image_path in image_paths]

    # Load chest classifications CSV file.
    print(f"Loading chest classifications CSV file {args.chest_classifications_csv_path}")
    with open(args.chest_classifications_csv_path, "r") as f:
        filtered_lines = [line for line in f if ";" in line]
    chest_classifications_df = pd.read_csv(StringIO("".join(filtered_lines)), low_memory=False, header=None, sep=";")

    # Create chest classifications dict.
    print("Creating chest classifications dict")
    chest_classifications_dict = dict(zip(chest_classifications_df[0], chest_classifications_df[1]))

    # Assign chest classifications.
    chest_classifications = []
    print("Assigning chest classifications")
    for image_path in image_paths:
        if image_path is None:
            chest_classifications.append(None)
        else:
            values = [chest_classifications_dict[item] if item in chest_classifications_dict else None for item in image_path]
            chest_classifications.append(values)

    # Update chest classification column.
    print("Updating chest classification column")
    assert len(chest_classifications) == len(df)
    df["chest_classification"] = chest_classifications

    # Save output.
    print(f"Saving output file {args.output_csv_path}")
    df.to_csv(args.output_csv_path, index=False)


def main(args):
    assign_chest_classifications(args)


if __name__ == "__main__":
    INPUT_CSV_PATH = "/mnt/all-data/reports/segmed/batch1/segmed_batch_1_merged_reports_with_image_paths_filtered_standardized_with_dicom_data_mapped_modalities_mapped_body_parts_with_uncertain_labels_cleaned_unflagged.csv"
    CHEST_CLASSIFICATIONS_CSV_PATH = "/mnt/all-data/reports/segmed/batch1/chest_non_chest_classificaton_results.csv"
    IMAGE_PATHS_COLUMN_NAME = "filtered_image_paths"
    OUTPUT_CSV_PATH = "/mnt/all-data/reports/segmed/batch1/segmed_batch_1_tmp.csv"

    args = argparse.Namespace(input_csv_path=INPUT_CSV_PATH,
                              chest_classifications_csv_path=CHEST_CLASSIFICATIONS_CSV_PATH,
                              image_paths_column_name=IMAGE_PATHS_COLUMN_NAME,
                              output_csv_path=OUTPUT_CSV_PATH)

    main(args)
