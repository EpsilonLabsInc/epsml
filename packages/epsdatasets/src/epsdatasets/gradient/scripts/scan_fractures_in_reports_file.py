import ast
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.visualization import visualization_utils

GCS_REPORTS_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT_CR_batch_1_chest_with_image_paths_with_fracture_labels_structured.csv"

fracture_labels_dist = {}


def row_handler(row, index):
    labels = row["fracture_labels_structured"]
    labels = ast.literal_eval(labels)

    for label in labels:
        fracture_type = label["fracture_type"]
        body_part = label["body_part"]

        if fracture_type not in fracture_labels_dist:
            fracture_labels_dist[fracture_type] = {}

        if body_part not in fracture_labels_dist[fracture_type]:
            fracture_labels_dist[fracture_type][body_part] = 1
        else:
            fracture_labels_dist[fracture_type][body_part] += 1


def main():
    print(f"Downloading reports file {GCS_REPORTS_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Loading reports file")
    df = pd.read_csv(StringIO(content))

    print("Reading reports file")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        row_handler(row, index)

    print("Fracture labels distribution:")
    print(fracture_labels_dist)

    for fracture_type in fracture_labels_dist:
        fig, _ = visualization_utils.generate_histogram(data=fracture_labels_dist[fracture_type],
                                                        title=fracture_type,
                                                        x_label="Body part",
                                                        y_label="Count",
                                                        x_labels_rotation_angle=45)
        file_name = fracture_type.lower().replace("/", "_")
        fig.savefig(f"{file_name}.png")


if __name__ == "__main__":
    main()
