import ast
import os
from enum import Enum

from epsutils.csv import csv_utils
from epsutils.gcs import gcs_utils


def get_gradient_instances_path(patient_id, accession_number, study_instance_uid, series_instance_uid):
    return os.path.join(patient_id, accession_number, "studies", study_instance_uid, "series", series_instance_uid, "instances")


def get_nifti_file_name(patient_id, accession_number, study_instance_uid, series_instance_uid):
    return gradient_instances_path_to_nifti_file_name(get_gradient_instances_path(
        patient_id=patient_id, accession_number=accession_number, study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid))


def get_volume_info_file_name(patient_id, accession_number, study_instance_uid, series_instance_uid):
    return gradient_instances_path_to_volume_info_file_name(get_gradient_instances_path(
        patient_id=patient_id, accession_number=accession_number, study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid))


def gradient_instances_path_to_nifti_file_name(instances_path):
    return instances_path.replace('/', '_') + ".nii.gz"


def gradient_instances_path_to_volume_info_file_name(instances_path):
    return instances_path.replace('/', '_') + ".txt"


def nifti_file_name_to_gradient_instances_path(nifti_file_name):
    assert nifti_file_name.endswith(".nii.gz")
    return nifti_file_name[:-7].replace("_", "/")  # Get rid of '.nii.gz' and replace slashes with underscores.


class BodyPartType(Enum):
    DICOM = 1
    GPT = 2


def get_all_body_parts_from_report(reports_file_path, gcs_bucket_name=None, body_part_type: BodyPartType=BodyPartType.GPT):
    if gcs_bucket_name is None:
        report = reports_file_path
    else:
        print(f"Downloading reports file from the {gcs_bucket_name} GCS bucket")
        report = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_bucket_name, gcs_file_name=reports_file_path)

    print("Loading reports file")
    column_name = "BodyPart" if body_part_type == BodyPartType.GPT else "BodyPartExamined"
    all_values = csv_utils.get_all_values(csv_file_name=report, column_name=column_name)
    all_values = [ast.literal_eval(value) for value in all_values if isinstance(value, str)]

    return all_values


def get_unique_body_parts_from_report(reports_file_path, gcs_bucket_name=None, body_part_type: BodyPartType=BodyPartType.GPT):
    if gcs_bucket_name is None:
        report = reports_file_path
    else:
        print(f"Downloading reports file from the {gcs_bucket_name} GCS bucket")
        report = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_bucket_name, gcs_file_name=reports_file_path)

    print("Loading reports file")
    column_name = "BodyPart" if body_part_type == BodyPartType.GPT else "BodyPartExamined"
    lists = csv_utils.get_unique_values(csv_file_name=report, column_name=column_name)

    unique_body_parts = set()
    for sublist in lists:
        try:
            unique_body_parts.update(ast.literal_eval(sublist))
        except Exception as e:
            print(f"Got this error when parsing body part '{sublist}': {str(e)}")

    unique_body_parts = list(unique_body_parts)

    return unique_body_parts
