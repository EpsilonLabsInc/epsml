import ast
import json

from tqdm import tqdm

MIMIC_TRAINING_FILE = "/home/andrej/tmp/interview/01.data_split/fold_1_train.json"
MIMIC_VALIDATION_FILE = "/home/andrej/tmp/interview/01.data_split/fold_1_val.json"
TARGET_LABEL = "Edema"
OUTPUT_TRAINING_FILE = "mimic-edema-training-bowen-split.jsonl"
OUTPUT_VALIDATION_FILE = "mimic-edema-validation-bowen-split.jsonl"


def main():
    # Load Mimic training file.

    print("Loading Mimic training file")

    with open(MIMIC_TRAINING_FILE, "r") as file:
        input_training_data = json.load(file)

    # Load mimic validation file.

    print("Loading Mimic validation file")

    with open(MIMIC_VALIDATION_FILE, "r") as file:
        input_validation_data = json.load(file)

    # Generate output training data.

    print("Generating output training data")

    output_training_data = []

    for input_sample in tqdm(input_training_data, total=len(input_training_data), desc="Processing"):
        image_paths = ast.literal_eval(input_sample["image_paths"])
        report_text = input_sample["cleaned_report_text"]
        pathologies = ast.literal_eval(input_sample["pathologies"])

        label = [TARGET_LABEL] if TARGET_LABEL in pathologies else []

        for image_path in image_paths:
            output_training_data.append({"image_path": image_path, "report_text": report_text, "labels": label, "original_labels": pathologies})

    # Generate output validation data.

    print("Generating output validation data")

    output_validation_data = []

    for input_sample in tqdm(input_validation_data, total=len(input_validation_data), desc="Processing"):
        image_paths = ast.literal_eval(input_sample["image_paths"])
        report_text = input_sample["cleaned_report_text"]
        pathologies = ast.literal_eval(input_sample["pathologies"])

        label = [TARGET_LABEL] if TARGET_LABEL in pathologies else []

        for image_path in image_paths:
            output_validation_data.append({"image_path": image_path, "report_text": report_text, "labels": label, "original_labels": pathologies})

    # Print statistics.

    num_positive = num_negative = 0

    for sample in output_training_data:
        if sample["labels"] == [TARGET_LABEL]:
            num_positive += 1
        else:
            num_negative += 1

    print(f"Num {TARGET_LABEL} labels in output training data = {num_positive} / {len(output_training_data)}")

    num_positive = num_negative = 0

    for sample in output_validation_data:
        if sample["labels"] == [TARGET_LABEL]:
            num_positive += 1
        else:
            num_negative += 1

    print(f"Num {TARGET_LABEL} labels in output validation data = {num_positive} / {len(output_validation_data)}")

    # Save output files.

    print("Saving output files")

    with open(OUTPUT_TRAINING_FILE, "w") as f:
        for item in output_training_data:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in output_validation_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
