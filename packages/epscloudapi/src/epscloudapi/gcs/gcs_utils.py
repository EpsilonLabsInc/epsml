import os
import time

from google.cloud import storage


def split_gcs_uri(gcs_uri):
    scheme = "gs://"

    if not gcs_uri.startswith(scheme):
        raise ValueError(f"GCS URI {gcs_uri} is missing {scheme} prefix")

    parts = gcs_uri.replace(scheme, "").split("/", 1)
    return {"scheme:": scheme, "gcs_bucket_name": parts[0], "gcs_path": parts[1]}


def is_gcs_uri(gcs_uri):
    try:
        gcs_data = split_gcs_uri(gcs_uri)
        return True
    except:
        return False


def list_files(gcs_bucket_name, gcs_dir, max_results=None, recursive=False):
    if not gcs_dir.endswith("/"):
        gcs_dir += "/"

    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_dir, max_results=max_results)

    return [
        f"gs://{gcs_bucket_name}/{blob.name}"
        for blob in blobs
        if recursive or "/" not in blob.name[len(gcs_dir):]
    ]


def download_file(gcs_bucket_name, gcs_file_name, local_file_name, num_retries=0, show_warning_on_retry=False):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_name)
    blob.download_to_filename(local_file_name)

    retry_count = 0
    while num_retries is None or retry_count < num_retries:
        if os.path.exists(local_file_name):
            break

        if show_warning_on_retry:
            print(f"Downloading of '{gcs_file_name}' from the '{gcs_bucket_name}' GCS bucket failed, retrying in 1 sec")

        time.sleep(1)
        blob.download_to_filename(local_file_name)
        retry_count += 1


def download_file_as_string(gcs_bucket_name, gcs_file_name):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_name)
    content = blob.download_as_text()
    return content


def download_file_as_bytes(gcs_bucket_name, gcs_file_name):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_name)
    content = blob.download_as_bytes()
    return content


def delete_file(gcs_bucket_name, gcs_file_name):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_name)
    blob.delete()


def upload_file(local_file_name, gcs_bucket_name, gcs_file_name):
    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_file_name)  # Destination name of the file.
        blob.upload_from_filename(local_file_name)  # Local file to be uploaded.
        return True, ""
    except Exception as e:
        return False, str(e)


def upload_file_stream(file_stream, gcs_bucket_name, gcs_file_name):
    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_file_name)  # Destination name of the file.
        blob.upload_from_file(file_stream)  # File stream to be uploaded.
        return True, ""
    except Exception as e:
        return False, str(e)


def upload_files(upload_data, gcs_bucket_name):
    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)

        for item in upload_data:
            blob = bucket.blob(item["gcs_file_name"])  # Destination name of the file.
            if item["is_file"]:
                blob.upload_from_filename(item["local_file_or_string"])  # Local file to be uploaded.
            else:
                blob.upload_from_string(item["local_file_or_string"])  # String to be uploaded.

        return True, ""
    except Exception as e:
        return False, str(e)


def check_if_file_exists(gcs_bucket_name, gcs_file_name):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_name)
    return blob.exists()


def check_if_files_exist(gcs_bucket_name, gcs_file_names):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    return all(bucket.blob(gcs_file_name).exists() for gcs_file_name in gcs_file_names)
