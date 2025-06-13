import argparse
import ast
import os

import pandas as pd
from tqdm import tqdm


def assign_projection_classifications(args):
    # Load input CSV file.
    print(f"Loading input CSV file {args.input_csv_path}")
    df = pd.read_csv(args.input_csv_path, low_memory=False)

    # Update image paths.
    print("Updating image paths")
    image_paths = df["image_paths"].tolist()
    base_paths = df["base_path"].tolist()
    for i, (image_path, base_path) in tqdm(enumerate(zip(image_paths, base_paths)), total=len(image_paths), desc="Processing"):
        image_path = ast.literal_eval(image_path)
        subst = args.base_path_substitutions[base_path]
        if subst is not None:
            image_paths[i] = [os.path.join(subst, item) for item in image_path]
        else:
            image_paths[i] = None

    # Load projection classifications CSV file.
    print(f"Loading projection classifications CSV file {args.projection_classifications_csv_path}")
    projection_classifications_df = pd.read_csv(args.projection_classifications_csv_path, low_memory=False, header=None)

    # Create projection classifications dict.
    print("Creating projection classifications dict")
    projection_classifications_dict = dict(zip(projection_classifications_df[0], projection_classifications_df[1]))

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
    print(f"Saving output file {args.output_file_path}")
    df.to_csv(args.output_file_path, index=False)


def main(args):
    assign_projection_classifications(args)


if __name__ == "__main__":
    INPUT_CSV_PATH = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"
    PROJECTION_CLASSIFICATIONS_CSV_PATH = "/home/andrej/tmp/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all_projections.csv"
    OUTPUT_FILE_PATH = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all_with_projections.csv"
    BASE_PATH_SUBSTITUTIONS = {
        "gradient/22JUL2024": "/mnt/efs/all-cxr/gradient/22JUL2024",
        "gradient/20DEC2024": "/mnt/efs/all-cxr/gradient/20DEC2024/deid",
        "gradient/09JAN2025": "/mnt/efs/all-cxr/gradient/09JAN2025/deid",
        "gradient/16AUG2024": None,
        "gradient/13JAN2025": None,
        "segmed/batch1": "/mnt/efs/all-cxr/segmed/batch1",
        "segmed/batch2": "/mnt/efs/all-cxr/segmed/batch2",
        "segmed/batch3": "/mnt/efs/all-cxr/segmed/batch3",
        "segmed/batch4": "/mnt/efs/all-cxr/segmed/batch4",
        "simonmed": "/mnt/efs/all-cxr/simonmed/images/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
    }

    args = argparse.Namespace(input_csv_path=INPUT_CSV_PATH,
                              projection_classifications_csv_path=PROJECTION_CLASSIFICATIONS_CSV_PATH,
                              output_file_path=OUTPUT_FILE_PATH,
                              base_path_substitutions=BASE_PATH_SUBSTITUTIONS)

    main(args)
