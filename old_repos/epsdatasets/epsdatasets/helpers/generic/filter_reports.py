import argparse
import ast

import pandas as pd
from tqdm import tqdm


def filter_reports(args):
    print(f"Loading input CSV file {args.input_csv_path}")
    df = pd.read_csv(args.input_csv_path, low_memory=False)
    print(f"Original dataset has {len(df)} rows")

    selected_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesing"):
        image_paths = ast.literal_eval(row["image_paths"])

        if args.apply_chest_filtering:
            body_part = row["body_part"]
            if body_part.strip().lower() != "chest":
                continue

            if pd.isna(row["chest_classification"]):
                continue

            chest_classification = ast.literal_eval(row["chest_classification"])
            assert len(chest_classification) == len(image_paths)

            if not all(elem.strip().lower() == "chest" for elem in chest_classification):
                continue

        if args.apply_projection_filtering:
            if pd.isna(row["projection_classification"]):
                continue

            projection_classification = ast.literal_eval(row["projection_classification"])
            assert len(projection_classification) == len(image_paths)

            if "Frontal" not in projection_classification or "Lateral" not in projection_classification:
                continue

            # Keep only the first frontal and the first lateral image.
            frontal_index = next((i for i, elem in enumerate(projection_classification) if elem == "Frontal"), None)
            lateral_index = next((i for i, elem in enumerate(projection_classification) if elem == "Lateral"), None)
            assert frontal_index is not None and lateral_index is not None
            row["image_paths"] = [image_paths[frontal_index], image_paths[lateral_index]]

        selected_rows.append(row)

    df = pd.DataFrame(selected_rows)
    print(f"Filtered dataset has {len(df)} rows")

    print(f"Saving output file {args.output_file_path}")
    df.to_csv(args.output_file_path, index=False)


def main(args):
    filter_reports(args)


if __name__ == "__main__":
    INPUT_CSV_PATH = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_test.csv"
    APPLY_CHEST_FILTERING = False
    APPLY_PROJECTION_FILTERING = True
    OUTPUT_FILE_PATH = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_chest_only_frontal_lateral_test.csv"

    args = argparse.Namespace(input_csv_path=INPUT_CSV_PATH,
                              apply_chest_filtering=APPLY_CHEST_FILTERING,
                              apply_projection_filtering=APPLY_PROJECTION_FILTERING,
                              output_file_path=OUTPUT_FILE_PATH)

    main(args)
