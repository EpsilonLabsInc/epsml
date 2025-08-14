import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm


def extract_tar_archive(tar_path, delete_after_extraction=False):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=os.path.dirname(tar_path))

    if delete_after_extraction and os.path.exists(tar_path):
        os.remove(tar_path)


def extract_tar_archives(root_dir, max_workers=8, delete_after_extraction=False):
    tar_paths = []

    print("Searching for TAR archives")

    for foldername, _, filenames in os.walk(root_dir):
        tar_paths.extend(os.path.join(foldername, f) for f in filenames if f.endswith(".tar"))

    print("Extracting TAR archives")

    extract_with_delete = partial(extract_tar_archive, delete_after_extraction=delete_after_extraction)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(extract_with_delete, tar_paths), total=len(tar_paths), desc="Processing", unit="file"))
