import argparse
import ast

import pandas as pd
from tqdm import tqdm


def row_handler(row):
    image_paths = ast.literal_eval(row["image_paths"])
    assert isinstance(image_paths, list)

    filtered_image_paths = ast.literal_eval(row["filtered_image_paths"])
    assert isinstance(filtered_image_paths, list)

    return {"num_images": len(image_paths), "num_accepted_images": len(filtered_image_paths)}


def main(args):
    print(f"Loading file {args.results_csv_file}")

    df = pd.read_csv(args.results_csv_file)

    print("Runnning analysis")

    num_all_studies = 0
    num_rejected_studies = 0
    num_all_images = 0
    num_accepted_images = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        res = row_handler(row)

        num_all_studies += 1

        if res["num_accepted_images"] == 0:
            num_rejected_studies += 1

        num_all_images += res["num_images"]
        num_accepted_images += res["num_accepted_images"]

    rejected_studies_ratio = num_rejected_studies / num_all_studies * 100
    num_rejected_images = num_all_images - num_accepted_images
    rejected_images_ratio = num_rejected_images / num_all_images * 100

    print(f"Number of rejected studies (studies with all images filtered out): {num_rejected_studies} ({rejected_studies_ratio:.2f}%)")
    print(f"Number of all studies: {num_all_studies}")
    print(f"Number of rejected images: {num_rejected_images} ({rejected_images_ratio:.2f}%)")
    print(f"Number of all images: {num_all_images}")


if __name__ == "__main__":
    RESULTS_CSV_FILE = "/mnt/efs/all-cxr/simonmed/batch1/simonmed_batch_1_reports_with_image_paths_filtered.csv"

    args = argparse.Namespace(results_csv_file=RESULTS_CSV_FILE)

    main(args)
