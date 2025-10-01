import ast
import os

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from epsutils.dicom import dicom_utils

GENERATED_DATA_FILE = "gradient-ct-16AGO2024-generated_data.csv"
GENERATED_DATA_FILE_NIFTI = "gradient-ct-16AGO2024-generated_data_nifti.csv"
GCS_BUCKET_NAME = "epsilon-data-us-central1"
GCS_IMAGES_DIR = "GRADIENT-DATABASE/CT/16AGO2024"
TEMP_DICOM_FILE = "temp.dcm"


print(f"Loading generated data from '{GENERATED_DATA_FILE}'")
df = pd.read_csv(GENERATED_DATA_FILE)
df = df.map(ast.literal_eval)  # Make sure all the elements are converted from strings back to original Python types.
volumes = df["volume"].to_dict()
volumes = {volume["path"]: volume["dicom_files"][0] for volume in volumes.values()}

print(f"Loading NIfTI generated data from '{GENERATED_DATA_FILE_NIFTI}'")
df_nifti = pd.read_csv(GENERATED_DATA_FILE_NIFTI)
df_nifti = df_nifti.map(ast.literal_eval)  # Make sure all the elements are converted from strings back to original Python types.
nifti_volumes = df_nifti["volume"].to_dict()
nifti_volumes = [volume["nifti_file"] for volume in nifti_volumes.values()]
num_nifti_volumes = len(nifti_volumes)

client = storage.Client()
bucket = client.bucket(GCS_BUCKET_NAME)

for nifti_volume in tqdm(nifti_volumes, total=num_nifti_volumes, desc="Processing"):
    nifti_base_name = os.path.splitext(os.path.splitext(nifti_volume)[0])[0]  # Remove .nii.gz extension.
    path = nifti_base_name.replace("_", "/")

    if path not in volumes:
        raise ValueError(f"Path '{path}' not in volumes")

    dicom_file = os.path.join(GCS_IMAGES_DIR, path, volumes[path])
    blob = bucket.blob(dicom_file)
    blob.download_to_filename(TEMP_DICOM_FILE)

    if not os.path.exists(TEMP_DICOM_FILE):
        raise ValueError(f"DICOM file {dicom_file} not properly downloaded")

    values = dicom_utils.read_dicom_tags(TEMP_DICOM_FILE, [{"name": "InstanceNumber", "required": True}])
    instance_number = values["InstanceNumber"]

    # if instance_number != 1:
    #     print(f"Instance number {instance_number} != 1 for dicom file '{dicom_file}'")

    dicom_content = dicom_utils.read_all_dicom_tags(TEMP_DICOM_FILE)
    dicom_content = "\n".join(dicom_content)
    dicom_content_file = os.path.join("content", nifti_base_name + ".txt")
    with open(dicom_content_file, "w") as file:
        file.write(dicom_content)

    os.remove(TEMP_DICOM_FILE)
