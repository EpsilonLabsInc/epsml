import argparse
import ast
import json
import os
import re
from io import BytesIO

import pandas as pd
from tqdm import tqdm

from epsutils.aws import aws_s3_utils
from epsutils.gcs import gcs_utils


def main(args):
    if aws_s3_utils.is_aws_s3_uri(args.master_csv_file):
        print(f"Downloading master CSV file {args.master_csv_file}")
        aws_s3_data = aws_s3_utils.split_aws_s3_uri(args.master_csv_file)
        content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
        content = BytesIO(content)
    else:
        content = args.master_csv_file

    print(f"Loading master CSV file {args.master_csv_file}")
    df = pd.read_csv(args.master_csv_file, low_memory=False)

    print("Getting a list of prediction probs files")
    assert gcs_utils.is_gcs_uri(args.prediction_probs_dir)
    gcs_data = gcs_utils.split_gcs_uri(args.prediction_probs_dir)
    all_files = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"], recursive=True)
    pattern = re.compile(r"prediction_probs_.*\.jsonl$")
    prediction_probs_files = [file_name for file_name in all_files if pattern.search(file_name)]

    print("Merging per-study probs")
    for index, prediction_probs_file in enumerate(prediction_probs_files):
        print(f"{index + 1}/{len(prediction_probs_files)} Merging per-study probs for {prediction_probs_file}")

        gcs_data = gcs_utils.split_gcs_uri(prediction_probs_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        probs = {}
        for line in content.splitlines():
            item = json.loads(line)
            assert len(item["file_name"]) == 1
            probs[item["file_name"][0]] = item["prob"]

        merged_probs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            found_image_paths = []
            found_probs = []
            for image_path in ast.literal_eval(row["image_paths"]):
                if image_path in probs:
                    found_image_paths.append(image_path)
                    found_probs.append(probs[image_path])

            if len(found_image_paths) > 0:
                merged_probs.append({"image_paths": found_image_paths, "probs": found_probs})

        jsonl_buffer = BytesIO()
        for item in merged_probs:
            jsonl_buffer.write((json.dumps(item) + "\n").encode("utf-8"))
        jsonl_buffer.seek(0)

        dir_name, file_name = os.path.split(prediction_probs_file)
        per_study_prediction_probs_file = os.path.join(dir_name, f"per_study_{file_name}")

        print(f"Uploading per-study prediction probs file {per_study_prediction_probs_file}")
        gcs_data = gcs_utils.split_gcs_uri(per_study_prediction_probs_file)
        gcs_utils.upload_file_stream(file_stream=jsonl_buffer, gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])


if __name__ == "__main__":
    MASTER_CSV_FILE = "s3://epsilonlabs-datasets/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"
    PREDICTION_PROBS_DIR = "gs://epsilonlabs-models/intern-vit-classifier/non-chest"

    args = argparse.Namespace(master_csv_file=MASTER_CSV_FILE,
                              prediction_probs_dir=PREDICTION_PROBS_DIR)

    main(args)
