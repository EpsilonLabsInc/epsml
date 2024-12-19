import logging
import os
from concurrent.futures import ProcessPoolExecutor
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

import config
from process_row import process_row, process_row_cr


def main():
    raise NotImplementedError("TODO: Do not use this script until the latest DICOM to NIfTI conversion is integrated.")

    raise NotImplementedError("TODO: Do not use this script until multi-frame DICOM removal is implemented.")

    # Configure logger.
    logging_utils.configure_logger(logger_file_name=os.path.join(config.OUTPUT_DIR, config.DISPLAY_NAME + "-log.txt"))

    # Show configuration settings.
    print(config.dump_config())
    logging.info(config.dump_config())

    # Download reports file.
    print(f"Downloading reports file from the GCS bucket into memory (in case of out-of-memory or killed error consider downloading to disk)")
    reports_file_content = gcs_utils.download_file_as_string(gcs_bucket_name=config.SOURCE_GCS_BUCKET_NAME, gcs_file_name=config.SOURCE_GCS_REPORTS_FILE)

    # Read reports file.
    print("Reading reports file")
    df = pd.read_csv(StringIO(reports_file_content), sep=",", low_memory=False)

    # Process rows in parallel.
    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        if "CR" in config.MODALITIES:
            results = list(tqdm(executor.map(process_row_cr, [row for _, row in df.iterrows()]), total=len(df), desc="Processing"))
        else:
            results = list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=len(df), desc="Processing"))


if __name__ == "__main__":
    main()
