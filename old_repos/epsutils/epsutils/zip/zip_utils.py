import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def extract_zip_archive(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))


def extract_zip_archives(root_dir, max_workers=8):
    zip_paths = []

    print("Searching for ZIP archives")

    for foldername, _, filenames in os.walk(root_dir):
        zip_paths.extend(os.path.join(foldername, f) for f in filenames if f.endswith(".zip"))

    print("Extracting ZIP archives")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(extract_zip_archive, zip_paths), total=len(zip_paths), desc="Processing", unit="file"))
