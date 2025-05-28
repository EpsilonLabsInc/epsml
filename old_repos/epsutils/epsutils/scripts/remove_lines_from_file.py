import argparse


def main(args):
    with open(args.input_file_path, "r") as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if args.line_content_to_remove not in line]

    with open(args.output_file_path, "w") as file:
        file.writelines(filtered_lines)


if __name__ == "__main__":
    INPUT_FILE_PATH = "/mnt/efs/all-cxr/simonmed/batch1/simonmed_batch_1_reports_with_image_paths_filtered.csv_20250527_121901.log"
    LINE_CONTENT_TO_REMOVE = "WARNING: decode_data_sequence is deprecated and will be removed in v4.0"
    OUTPUT_FILE_PATH = "/mnt/efs/all-cxr/simonmed/batch1/simonmed_batch_1_reports_with_image_paths_filtered.csv_20250527_121901_new.log"

    args = argparse.Namespace(input_file_path=INPUT_FILE_PATH,
                              line_content_to_remove=LINE_CONTENT_TO_REMOVE,
                              output_file_path=OUTPUT_FILE_PATH)

    main(args)
