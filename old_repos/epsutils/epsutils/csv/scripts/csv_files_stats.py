import argparse

import pandas as pd


def main(args):
    stats_dict = {
        "num_rows": 0
    }

    for index, csv_file in enumerate(args.csv_files):
        print(f"{index + 1}/{len(args.csv_files)} Loading {csv_file}")
        df = pd.read_csv(csv_file, low_memory=False)
        stats_dict["num_rows"] += len(df)

    print("Statistics:")
    for key, value in stats_dict.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    CSV_FILES = [
        "/mnt/sfs-segmed-1/reports/CO2_588_Batch_1_Part_1_delivered_studies.csv",
        "/mnt/sfs-segmed-2/reports/CO2-658_part2.csv",
        "/mnt/sfs-segmed-34/segmed_3/reports/CO2_354_2_2_June_2025_deliveries.csv",
        "/mnt/sfs-segmed-34/segmed_4/reports/CO2_588_Batch_1_Part_3_delivered_studies.csv"
    ]

    args = argparse.Namespace(csv_files=CSV_FILES)

    main(args)
