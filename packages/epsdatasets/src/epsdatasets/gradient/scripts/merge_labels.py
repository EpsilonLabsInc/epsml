from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_REPORTS_FILE_1 = "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_alveolar_expanded_labels.csv"
LABEL_COLUMN_NAME_1 = "alveolar_expanded_labels"
GCS_REPORTS_FILE_2 = "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_cardio_labels.csv"
LABEL_COLUMN_NAME_2 = "cardio_labels"


def main():
    # Download reports file #1.
    print("Downloading reports file #1")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE_1)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Read reports file #1.
    print("Reading reports file #1")
    df1 = pd.read_csv(StringIO(content), low_memory=False)

    # Download reports file #2.
    print("Downloading reports file #2")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE_2)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Read reports file #2.
    print("Reading reports file #2")
    df2 = pd.read_csv(StringIO(content), low_memory=False)

    # Generate merged dataset.
    print("Generating merged dataset")
    merged_df = df1.copy()

    # Merge labels.
    print("Merging labels")
    merged_df[LABEL_COLUMN_NAME_2] = None
    merged_df["merged_labels"] = None

    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing"):
        try:
            curr_labels = [label.strip() for label in row[LABEL_COLUMN_NAME_1].split(",")]

            labels1 = df1.loc[index, LABEL_COLUMN_NAME_1]
            labels1 = [label.strip() for label in labels1.split(",")]

            labels2 = df2.loc[index, LABEL_COLUMN_NAME_2]
            labels2 = [label.strip() for label in labels2.split(",")]
        except Exception as e:
            print(f"Error reading labels: {str(e)}")
            continue

        assert curr_labels == labels1

        merged_labels = labels1 + labels2
        merged_labels = list(set(merged_labels))  # Remove duplicates.

        # 'No Findings' in combination with any other pathology is not 'No Findings' anymore.
        if len(merged_labels) > 1 and "No Findings" in merged_labels:
            merged_labels.remove("No Findings")

        merged_df.at[index, LABEL_COLUMN_NAME_2] = ", ".join(labels2)
        merged_df.at[index, "merged_labels"] = ", ".join(merged_labels)

    # Save merged dataset.
    print("Saving merged dataset")
    file_name = "GRADIENT_CR_ALL_CHEST_BATCHES_merged_cleaned_alveolar_expanded_labels_and_cardio_labels.csv"
    merged_df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
