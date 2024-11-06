from google.cloud import storage
from tqdm import tqdm


GCS_BUCKET_NAME = "gradient-mrs-nifti"
GCS_IMAGES_DIR = "01OCT2024/"  # Dir must end with a slash in order for the code below to work correctly (i.e., not to look for files recursively)!

client = storage.Client()
bucket = client.bucket(GCS_BUCKET_NAME)
blobs = bucket.list_blobs(prefix=GCS_IMAGES_DIR)

print("Generating a set of all .nii.gz files and a set of all .txt files")

nii_files = set()
txt_files = set()
progress_bar = tqdm(unit=" files", leave=False)

for blob in blobs:
    progress_bar.update(1)

    # Avoid recursion.
    if "/" in blob.name[len(GCS_IMAGES_DIR):]:
        continue

    if blob.name.endswith('.nii.gz'):
        nii_files.add(blob.name[:-7])
    elif blob.name.endswith('.txt'):
        txt_files.add(blob.name[:-4])

progress_bar.close()

print(f"Number of .nii.gz files: {len(nii_files)}")
print(f"Number of .txt files: {len(txt_files)}")

print("Looking for inconsistent file pairs")

inconsistent_files = nii_files - txt_files

if inconsistent_files:
    print(f"Inconsistent file pairs: {inconsistent_files}")
else:
    print("All .nii.gz files have corresponding .txt files")
