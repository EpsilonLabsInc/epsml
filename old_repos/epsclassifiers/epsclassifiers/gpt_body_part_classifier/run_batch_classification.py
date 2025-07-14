import argparse
import ast
import json
import os
from io import StringIO, BytesIO

import pandas as pd
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.gpt import gpt_utils
from epsutils.image import image_utils


def main(args):
    if gcs_utils.is_gcs_uri(args.reports_file):
        print(f"Downloading reports file {args.reports_file}")
        gcs_data = gcs_utils.split_gcs_uri(args.reports_file)
        content = StringIO(gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
    else:
        content = args.reports_file

    print("Loading reports file")
    df = pd.read_csv(content, low_memory=False)

    print("Reading reports file")
    request_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        base_path = row["base_path"]
        if base_path not in args.base_path_substitutions:
            raise ValueError(f"Base path '{base_path}' not in base path substitutions")
        elif args.base_path_substitutions[base_path] is None:
            continue
        subst = args.base_path_substitutions[base_path]

        dicom_body_part = row["body_part_dicom"].lower() if pd.notna(row["body_part_dicom"]) else ""
        if args.dicom_body_part not in dicom_body_part:
            continue

        images = []
        for image_path in ast.literal_eval(row["image_paths"]):
            image_path = os.path.join(subst, image_path)
            image = dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
            image = image.resize(args.target_image_size)

            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            images.append(buffer)

        request = gpt_utils.create_request(prompt="Which body part is in these x-ray images?",
                                           images=images,
                                           request_id=str(request_id),
                                           deployment=args.gpt_config["batch_deployment"])

        content = json.dumps(request, ensure_ascii=False) + "\n"

        content = gpt_utils.run_batch(input_jsonl=content,
                                      endpoint=args.gpt_config["endpoint"],
                                      api_key=args.gpt_config["api_key"],
                                      api_version=args.gpt_config["api_version"],
                                      is_content=True)

        # TODO: Remove.
        print(content)
        exit()

        request_id += 1


if __name__ == "__main__":
    REPORTS_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_test.csv"  # Reports CSV file. Can be local file or GCS URI.
    DICOM_BODY_PART = "arm"
    BASE_PATH_SUBSTITUTIONS = {
        "gradient/22JUL2024": None,
        "gradient/20DEC2024": None,
        "gradient/09JAN2025": None,
        "gradient/16AUG2024": "/mnt/sfs-gradient-nochest/16AUG2024",
        "gradient/13JAN2025": "/mnt/sfs-gradient-nochest/13JAN2025/deid",
        "segmed/batch1": "/mnt/sfs-segmed-1",
        "segmed/batch2": "/mnt/sfs-segmed-2",
        "segmed/batch3": "/mnt/sfs-segmed-34/segmed_3",
        "segmed/batch4": "/mnt/sfs-segmed-34/segmed_4",
        "simonmed": "/mnt/sfs-simonmed"
    }
    TARGET_IMAGE_SIZE = (200, 200)
    GPT_CONFIG = {
        "endpoint": "https://epsilon-eastus.openai.azure.com/",
        "api_key": "9b568fdffb144272811cb5fad8b584a0",
        "api_version": "2024-12-01-preview",
        "batch_deployment": "gpt-4.1",
        "batch_mini_deployment": "gpt-4.1"
    }

    args = argparse.Namespace(reports_file=REPORTS_FILE,
                              dicom_body_part=DICOM_BODY_PART,
                              base_path_substitutions=BASE_PATH_SUBSTITUTIONS,
                              target_image_size=TARGET_IMAGE_SIZE,
                              gpt_config=GPT_CONFIG)

    main(args)
