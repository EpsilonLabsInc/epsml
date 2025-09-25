import argparse
import json
import os
import yaml
from pathlib import Path
from typing import List, Tuple

from epsutils.training.scheduler.training_job_analyzer import TrainingJobAnalyzer

EXPECTED_BATCH_SIZE = 128
EXPECTED_BATCH_TIME_IN_SEC = 10


def get_training_jobs(training_jobs_file_or_config_path):
    path = Path(training_jobs_file_or_config_path)

    if path.is_file() and path.suffix.lower() == ".json":
        print("Loading training jobs")
        with open(path, "r") as file:
            training_jobs = json.load(file)

    elif path.is_dir():
        print("Searching for config files and analyzing training jobs, this might take a while...")
        training_jobs = TrainingJobAnalyzer().find_config_files_and_get_training_jobs(path)

    else:
        raise ValueError("training_jobs_file_or_config_path must be either .json file or path to the config files")

    return training_jobs


def assign_training_jobs_to_buckets(training_jobs: List, num_buckets: int, min_epochs: int = 2, max_epochs: int = 10) -> List[List[Tuple[str, int]]]:
    print(f"Assigning {len(training_jobs)} training jobs to {num_buckets} buckets")

    assert any(job["training_dataset_size"] is not None for job in training_jobs)

    max_count = max(job["training_dataset_size"] for job in training_jobs if job["training_dataset_size"] is not None)
    max_total_count = min_epochs * max_count

    print("Determining number of epochs for each training job using the following values:")
    print(f"+ Max count: {max_count}")
    print(f"+ Max total count: {max_total_count}")
    print(f"+ Min epochs: {min_epochs}")
    print(f"+ Max epochs: {max_epochs}")

    total_counts = []
    for job in training_jobs:
        config_file = job["config_file"]
        count = job["training_dataset_size"]

        if count is None:
            continue

        num_epochs = max_total_count // count
        num_epochs = max(min(num_epochs, max_epochs), min_epochs)
        total_count = count * num_epochs
        total_counts.append({"config_file": config_file, "count": count, "num_epochs": num_epochs, "total_count": total_count})

    total_counts.sort(key=lambda x: x["total_count"], reverse=True)

    buckets = [[] for _ in range(num_buckets)]
    bucket_sums = [0] * num_buckets

    for item in total_counts:
        min_bucket_idx = bucket_sums.index(min(bucket_sums))
        buckets[min_bucket_idx].append(item)
        bucket_sums[min_bucket_idx] += item["total_count"]

    return buckets, bucket_sums


def update_num_epochs(input_config_file, output_config_file, num_epochs):
    with open(input_config_file, "r") as file:
        config = yaml.safe_load(file)

    config["training"]["num_epochs"] = num_epochs

    with open(output_config_file, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def copy_config_files(buckets, output_dir):
    print("Copying config files into buckets")

    for index, bucket in enumerate(buckets):
        bucket_dir = os.path.join(output_dir, f"bucket_{index + 1}")
        os.makedirs(bucket_dir, exist_ok=True)

        for item in bucket:
            update_num_epochs(input_config_file=item["config_file"],
                              output_config_file=os.path.join(bucket_dir, os.path.basename(item["config_file"])),
                              num_epochs=item["num_epochs"])


def generate_summary(buckets, bucket_sums):
    total_count = sum(bucket_sums)
    min_count = min(bucket_sums)
    max_count = max(bucket_sums)

    expected_times = []
    for bucket_sum in bucket_sums:
        expected_time = (bucket_sum / EXPECTED_BATCH_SIZE) * EXPECTED_BATCH_TIME_IN_SEC / 3600
        expected_time = round(expected_time, 2)
        expected_times.append(expected_time)

    min_time = min(expected_times)
    max_time = max(expected_times)

    print("")
    print("Summary:")
    print(f"+ Buckets created: {len(buckets)}")
    print(f"+ Total count over all buckets: {total_count:,}")
    print(f"+ Min count: {min_count:,}")
    print(f"+ Max count: {max_count:,}")
    print(f"+ Training jobs distribution: {[len(bucket) for bucket in buckets]}")
    print(f"+ Expected training times (h): {expected_times}")
    print(f"+ Min expected training time (h): {min_time}")
    print(f"+ Max expected training time (h): {max_time}")


def main(args):
    training_jobs = get_training_jobs(args.training_jobs_file_or_config_path)

    buckets, bucket_sums = assign_training_jobs_to_buckets(training_jobs=training_jobs,
                                                           num_buckets=args.num_buckets,
                                                           min_epochs=args.min_epochs,
                                                           max_epochs=args.max_epochs)

    copy_config_files(buckets=buckets, output_dir=args.output_dir)

    generate_summary(buckets=buckets, bucket_sums=bucket_sums)


if __name__ == "__main__":
    TRAINING_JOBS_FILE_OR_CONFIG_PATH = "training_jobs.json"
    NUM_BUCKETS = 45
    MIN_EPOCHS = 2
    MAX_EPOCHS = 10
    OUTPUT_DIR = "/mnt/training/v2.0.0/buckets"

    args = argparse.Namespace(training_jobs_file_or_config_path=TRAINING_JOBS_FILE_OR_CONFIG_PATH,
                              num_buckets=NUM_BUCKETS,
                              min_epochs=MIN_EPOCHS,
                              max_epochs=MAX_EPOCHS,
                              output_dir=OUTPUT_DIR)

    main(args)
