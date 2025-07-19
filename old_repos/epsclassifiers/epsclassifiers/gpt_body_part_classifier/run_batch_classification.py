import argparse
import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import StringIO, BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.gpt import gpt_utils
from epsutils.image import image_utils

import prompts


def main(args):
    if gcs_utils.is_gcs_uri(args.reports_file):
        print(f"Downloading reports file {args.reports_file}")
        gcs_data = gcs_utils.split_gcs_uri(args.reports_file)
        content = StringIO(gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
    else:
        content = args.reports_file

    print("Loading reports file")
    df = pd.read_csv(content, low_memory=False)

    print("Filtering reports")
    filtered_reports = filter_reports(df=df,
                                      base_path_substitutions=args.base_path_substitutions,
                                      target_dicom_body_parts=args.target_dicom_body_parts,
                                      target_image_size=args.target_image_size,
                                      use_png=args.use_png)

    if args.max_num_rows is not None:
        filtered_reports = filtered_reports[:args.max_num_rows]

    print("Filtered reports samples:")
    print(filtered_reports[:10])

    print("")
    print(f"Number of filtered reports: {len(filtered_reports)}")

    print("Running batches")
    input_file_names, output_file_names = run_batches(filtered_reports=filtered_reports, gpt_prompt=args.gpt_prompt, gpt_config=args.gpt_config, max_workers=args.max_workers)

    print("Assemble results")
    results = assemble_results(output_file_names)

    print("Updating reports")
    mapping = {item["index"]: item["result"] for item in results}
    df[args.column_name_to_add] = pd.Series(mapping).reindex(df.index)

    print("Saving output file")
    df.to_csv(args.output_file, index=False)

    if args.clean_up_files:
        print("Cleaning up files")
        files_to_delete = input_file_names + output_file_names
        for file_to_delete in files_to_delete:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

    if args.gpt_config["clean_up_azure_files"]:
        print("Cleaning up Azure files")
        gpt_utils.delete_files(endpoint=args.gpt_config["endpoint"],
                               api_key=args.gpt_config["api_key"],
                               api_version=args.gpt_config["api_version"],
                               purpose="batch")


def process_row(row, base_path_substitutions, target_dicom_body_parts, target_image_size, use_png):
    if pd.isna(row.report_text):
        return None

    # Get base path substitution.
    base_path = row.base_path
    if base_path not in base_path_substitutions:
        return None
    elif base_path_substitutions[base_path] is None:
        return None
    subst = base_path_substitutions[base_path]

    # Check body part DICOM tag. If it differs from target body parts, skip the row.
    dicom_body_part = row.body_part_dicom.lower() if pd.notna(row.body_part_dicom) else ""
    if not any(item in dicom_body_part for item in target_dicom_body_parts):
        return None

    # Get all study images and convert them to JPEG.
    images = []
    for image_path in ast.literal_eval(row.image_paths):
        image_path = os.path.join(subst, image_path)

        try:
            if use_png:
                image_path = image_path.replace(".dcm", ".png")
                image = Image.open(image_path)
            else:
                image = dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        except Exception as e:
            continue

        image = image.resize(target_image_size)

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        images.append(buffer)

    if images == []:
        return None

    return {
        "index": row.Index,
        "images": images,
        "report_text": row.report_text,
        "body_part_dicom": row.body_part_dicom
    }


def filter_reports(df, base_path_substitutions, target_dicom_body_parts, target_image_size, use_png):
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda row: process_row(row, base_path_substitutions, target_dicom_body_parts, target_image_size, use_png), [row for row in df.itertuples()]),
                            total=len(df),
                            desc="Processing"))

    filtered_reports = [item for item in results if item is not None]

    return filtered_reports


def run_batches(filtered_reports, gpt_prompt, gpt_config, max_workers):
    requests = []

    for item in filtered_reports:
        request = gpt_utils.create_request(system_prompt=gpt_prompt,
                                           user_prompt=item["report_text"],
                                           images=item["images"],
                                           request_id=str(item["index"]),
                                           deployment=gpt_config["batch_deployment"])
        requests.append(request)

    input_file_names = gpt_utils.save_requests_as_jsonl(requests=requests, file_name="input.jsonl")
    output_file_names = [input_file_name.replace("input", "output") for input_file_name in input_file_names]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, input_file_name in enumerate(input_file_names):
            futures.append(executor.submit(
                    gpt_utils.run_batch,
                    input_file_name,
                    output_file_names[index],
                    gpt_config["endpoint"],
                    gpt_config["api_key"],
                    gpt_config["api_version"]
                )
            )

        for future in as_completed(futures):
            future.result()

    return input_file_names, output_file_names


def assemble_results(output_file_names):
    results = []

    for output_file_name in output_file_names:
        with open(output_file_name, "r") as file:
            for line in file:
                data = json.loads(line)
                index = int(data["custom_id"])
                result = data["response"]["body"]["choices"][0]["message"]["content"].strip().strip("\"")
                results.append({"index": index, "result": result})

    return results


if __name__ == "__main__":
    REPORTS_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_train.csv"  # Reports CSV file. Can be local file or GCS URI.
    OUTPUT_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_with_arm_segment_train.csv"
    USE_PNG = True
    COLUMN_NAME_TO_ADD = "arm_segment"
    TARGET_DICOM_BODY_PARTS = prompts.ARM_SEGMENTS_TARGET_DICOM_BODY_PARTS
    TARGET_IMAGE_SIZE = (200, 200)
    MAX_NUM_ROWS = None
    MAX_WORKERS = 20
    CLEAN_UP_FILES = True
    BASE_PATH_SUBSTITUTIONS = {
        "gradient/22JUL2024": None,
        "gradient/20DEC2024": None,
        "gradient/09JAN2025": None,
        "gradient/16AUG2024": "/mnt/png/512x512/gradient/16AUG2024",
        "gradient/13JAN2025": "/mnt/png/512x512/gradient/13JAN2025/deid",
        "segmed/batch1": "/mnt/png/512x512/segmed/batch1",
        "segmed/batch2": "/mnt/png/512x512/segmed/batch2",
        "segmed/batch3": "/mnt/png/512x512/segmed/batch3",
        "segmed/batch4": "/mnt/png/512x512/segmed/batch4",
        "simonmed": "/mnt/png/512x512/simonmed"
    }
    GPT_PROMPT = prompts.ARM_SEGMENTS_GPT_PROMPT
    GPT_CONFIG = {
        "endpoint": "https://epsilon-eastus.openai.azure.com/",
        "api_key": "9b568fdffb144272811cb5fad8b584a0",
        "api_version": "2024-12-01-preview",
        "batch_deployment": "gpt-4.1",
        "batch_mini_deployment": "gpt-4.1",
        "standard_deployment": "gpt-4o-standard",
        "standard_mini_deployment": "gpt-4o-mini-standard",
        "clean_up_azure_files": True
    }

    args = argparse.Namespace(reports_file=REPORTS_FILE,
                              output_file=OUTPUT_FILE,
                              use_png=USE_PNG,
                              column_name_to_add=COLUMN_NAME_TO_ADD,
                              target_dicom_body_parts=TARGET_DICOM_BODY_PARTS,
                              target_image_size=TARGET_IMAGE_SIZE,
                              max_num_rows=MAX_NUM_ROWS,
                              max_workers=MAX_WORKERS,
                              clean_up_files=CLEAN_UP_FILES,
                              base_path_substitutions=BASE_PATH_SUBSTITUTIONS,
                              gpt_prompt=GPT_PROMPT,
                              gpt_config=GPT_CONFIG)

    main(args)
