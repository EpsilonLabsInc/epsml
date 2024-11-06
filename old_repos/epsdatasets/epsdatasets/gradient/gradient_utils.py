import os


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
