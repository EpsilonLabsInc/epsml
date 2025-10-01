import ast

import numpy as np
import pandas as pd
from tqdm import tqdm

from epsdatasets.helpers.gradient.gradient_labels import GRADIENT_LABELS
from epsutils.labels.labels_manager import LabelsManager

TRAINING_DATA_FILE = "/home/andrej/data/gradient/ct_chest_only_training_data.csv"
NUM_TOP = 4


def main():
    df = pd.read_csv(TRAINING_DATA_FILE)
    labels_manager = LabelsManager(all_label_names=GRADIENT_LABELS)
    all_label_names = labels_manager.get_all_label_names()
    print(f"All label names: {all_label_names}")

    stacked_labels = np.full((len(df), 21), np.nan)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        labels = np.array(ast.literal_eval(row["label"]))
        stacked_labels[index] = labels

    assert not np.any(np.isnan(stacked_labels))
    print(f"Stacked labels shape: {stacked_labels.shape}")

    ones_count_per_column = np.sum(stacked_labels, axis=0)
    assert len(ones_count_per_column) == len(all_label_names)

    # Print label counts.
    print("")
    print("--------------------")
    print("Label counts:")
    print("--------------------")
    for i in range(len(ones_count_per_column)):
        print(f"{labels_manager.get_label_name_at(i)}: {ones_count_per_column[i]}")

    # Print top label indices.
    print("")
    print("--------------------")
    print("Top label indices:")
    print("--------------------")
    top_indices = np.argsort(ones_count_per_column)[-NUM_TOP:][::-1]
    print(top_indices)

    # Print top label counts.
    print("")
    print("--------------------")
    print("Top label counts:")
    print("--------------------")
    for i in range(len(top_indices)):
        print(f"{labels_manager.get_label_name_at(top_indices[i])}: {ones_count_per_column[top_indices[i]]}")

if __name__ == "__main__":
    main()
