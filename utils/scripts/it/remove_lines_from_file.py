import argparse
from enum import Enum


class Mode(Enum):
    REMOVE_IF_INCLUDES = 1
    REMOVE_IF_NOT_INCLUDES = 2
    REMOVE_IF_STARTS_WITH = 3
    REMOVE_IF_NOT_STARTS_WITH = 4


def main(args):
    with open(args.input_file_path, "r") as file:
        lines = file.readlines()

    filtered_lines = []
    for line in lines:
        if args.mode == Mode.REMOVE_IF_INCLUDES:
            if args.content not in line:
                filtered_lines.append(line)
        elif args.mode == Mode.REMOVE_IF_NOT_INCLUDES:
            if args.content in line:
                filtered_lines.append(line)
        elif args.mode == Mode.REMOVE_IF_STARTS_WITH:
            if not line.startswith(args.content):
                filtered_lines.append(line)
        elif args.mode == Mode.REMOVE_IF_NOT_STARTS_WITH:
            if line.startswith(args.content):
                filtered_lines.append(line)
        else:
            raise ValueError(f"Unsupported mode {args.mode}")

    with open(args.output_file_path, "w") as file:
        file.writelines(filtered_lines)


if __name__ == "__main__":
    INPUT_FILE_PATH = "/home/andrej/tmp/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all_projections.csv"
    CONTENT = "/mnt/efs"
    MODE = Mode.REMOVE_IF_NOT_STARTS_WITH
    OUTPUT_FILE_PATH = "/home/andrej/tmp/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all_projections.csv"

    args = argparse.Namespace(input_file_path=INPUT_FILE_PATH,
                              content=CONTENT,
                              mode=MODE,
                              output_file_path=OUTPUT_FILE_PATH)

    main(args)
