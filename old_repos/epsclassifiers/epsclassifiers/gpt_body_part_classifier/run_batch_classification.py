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

from epsclassifiers.gpt_body_part_classifier import prompts

TARGET_IMAGE_SIZE = (200, 200)
MAX_WORKERS = 20  # Should not be more than 20 due to GPT API concurrency limitation.

GPT_CONFIG = {
    "endpoint": "https://epsilon-eastus.openai.azure.com/",
    "api_key": "9b568fdffb144272811cb5fad8b584a0",
    "api_version": "2024-12-01-preview",
    "batch_deployment": "gpt-4.1",
    "batch_mini_deployment": "gpt-4.1",
    "standard_deployment": "gpt-4o-standard",
    "standard_mini_deployment": "gpt-4o-mini-standard"
}


def process_row(row, image_paths_column, base_images_path, use_png):
    if pd.isna(row["report_text"]):
        return None

    image_paths = ast.literal_eval(row[image_paths_column])
    images = []

    # Get all study images and convert them to JPEG.
    for image_path in image_paths:
        image_path = os.path.join(base_images_path, image_path)

        try:
            if use_png:
                image_path = image_path.replace(".dcm", ".png")
                image = Image.open(image_path)
            else:
                image = dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        except Exception as e:
            print(str(e))
            continue

        image = image.resize(TARGET_IMAGE_SIZE)

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        images.append(buffer)

    if images == []:
        return None

    return {
        "index": row["index"],
        "images": images,
        "report_text": row["report_text"]
    }

def process_row_wrapper(args):
    row, image_paths_column, base_images_path, use_png = args
    return process_row(row, image_paths_column, base_images_path, use_png)

def extract_data(df, image_paths_column, base_images_path, use_png, per_image_classification):
    rows = [{**row, "index": idx} for idx, row in zip(df.index, df.to_dict(orient="records"))]
    args_list = [(row, image_paths_column, base_images_path, use_png) for row in rows]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row_wrapper, args_list),
                            total=len(df),
                            desc="Processing"))

    # Get rid of invalid items.
    data = [item for item in results if item is not None]

    # In case of per-image classification, data needs to be flattened so that each study image is a separate entry.
    if per_image_classification:
        data = [
            {
                "index": item["index"],
                "image_index": i,
                "images": [image],
                "report_text": item["report_text"],
            }
            for item in data
            for i, image in enumerate(item["images"])
        ]

    return data

def generate_request_id(item):
        if "image_index" in item:
            return f"{item['index']}_{item['image_index']}"
        else:
            return f"{item['index']}"

def run_batches(data, gpt_prompt):
    requests = []

    for item in data:
        request = gpt_utils.create_request(system_prompt=gpt_prompt,
                                           user_prompt=item["report_text"],
                                           images=item["images"],
                                           request_id=generate_request_id(item),
                                           deployment=GPT_CONFIG["batch_deployment"])
        requests.append(request)

    input_file_names = gpt_utils.save_requests_as_jsonl(requests=requests, file_name="input.jsonl")
    output_file_names = [input_file_name.replace("input", "output") for input_file_name in input_file_names]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for index, input_file_name in enumerate(input_file_names):
            futures.append(executor.submit(
                    gpt_utils.run_batch,
                    input_file_name,
                    output_file_names[index],
                    GPT_CONFIG["endpoint"],
                    GPT_CONFIG["api_key"],
                    GPT_CONFIG["api_version"]
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
                index = data["custom_id"]
                result = data["response"]["body"]["choices"][0]["message"]["content"].strip().strip("\"")
                try:
                    parsed_result = ast.literal_eval(result)
                except:
                    parsed_result = result
                results.append({"index": index, "result": parsed_result})

    return results

def aggregate_per_image_results_to_per_study(results):
    aggregated_results = {}

    for item in results:
        # Split index like "5_3" into study_index=5 and image_index=3
        study_index_str, image_index_str = item["index"].split("_")
        study_index = int(study_index_str)
        image_index = int(image_index_str)

        if study_index not in aggregated_results:
            aggregated_results[study_index] = []

        aggregated_results[study_index].append((image_index, item["result"]))

    # Sort each study's results by image_index and flatten the result list.
    results = [
        {
            "index": study_index,
            "result": [result for _, result in sorted(body_parts)]
        }
        for study_index, body_parts in aggregated_results.items()
    ]

    return results

def main(args):
    print("GPT body part classifier will be using the following configuration:")
    print(f"{json.dumps(vars(args), indent=4)}")

    if gcs_utils.is_gcs_uri(args.reports_file):
        print(f"Downloading reports file {args.reports_file}")
        gcs_data = gcs_utils.split_gcs_uri(args.reports_file)
        content = StringIO(gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
    else:
        content = args.reports_file

    print("Loading reports file")
    df = pd.read_csv(content, low_memory=False)

    print("Extracting data from reports")
    extracted_data = extract_data(df=df,
                                  image_paths_column=args.image_paths_column,
                                  base_images_path=args.base_images_path,
                                  use_png=args.use_png,
                                  per_image_classification=args.per_image_classification)

    print(f"Extracted data has {len(extracted_data)} rows")

    print("Running batches")
    input_file_names, output_file_names = run_batches(data=extracted_data, gpt_prompt=args.gpt_prompt)

    print("Assemble results")
    results = assemble_results(output_file_names)

    if args.per_image_classification:
        print("Aggregating per-image results back to per-study")
        results = aggregate_per_image_results_to_per_study(results)

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

    print("Cleaning up Azure files")
    gpt_utils.delete_files(endpoint=args.gpt_config["endpoint"],
                            api_key=args.gpt_config["api_key"],
                            api_version=args.gpt_config["api_version"],
                            force=True,
                            purpose="batch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-based body part classification on X-ray images & reports.")
    parser.add_argument("--reports_file", type=str, default="/mnt/all-data/reports/segmed/batch1/segmed_batch_1_final.csv", help="Path to the reports CSV file. Can be local file or GCS URI.")
    parser.add_argument("--output_file", type=str, default="/mnt/all-data/reports/segmed/batch1/segmed_batch_1_final_with_body_parts.csv", help="Path to the output CSV file.")
    parser.add_argument("--image_paths_column", type=str, default="image_paths", help="Column name containing image paths.")
    parser.add_argument("--base_images_path", type=str, default="/mnt/all-data/png/512x512/segmed/batch1", help="Base directory for image files.")
    parser.add_argument("--use_png", action="store_true", help="Use PNG images instead of DICOM?")
    parser.add_argument("--per_image_classification", action="store_true", help="Run per-image classification instead of per-study?")
    parser.add_argument("--column_name_to_add", type=str, default="body_part", help="Name of the column with GPT results to add.")
    parser.add_argument("--clean_up_files", action="store_true", help="Clean up local files after processing?")
    parser.add_argument("--gpt_prompt", type=str, default=prompts.ALL_BODY_PARTS_GPT_PROMPT, help="Custom GPT prompt string.")
    args = parser.parse_args()

    main(args)
