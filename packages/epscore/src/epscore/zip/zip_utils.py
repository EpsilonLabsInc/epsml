import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm


def extract_zip_archive(zip_path, delete_after_extraction=False):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))

    if delete_after_extraction and os.path.exists(zip_path):
        os.remove(zip_path)


def extract_zip_archives(root_dir, max_workers=8, delete_after_extraction=False):
    zip_paths = []

    print("Searching for ZIP archives")

    for foldername, _, filenames in os.walk(root_dir):
        zip_paths.extend(os.path.join(foldername, f) for f in filenames if f.endswith(".zip"))

    print("Extracting ZIP archives")

    extract_with_delete = partial(extract_zip_archive, delete_after_extraction=delete_after_extraction)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(extract_with_delete, zip_paths), total=len(zip_paths), desc="Processing", unit="file"))
