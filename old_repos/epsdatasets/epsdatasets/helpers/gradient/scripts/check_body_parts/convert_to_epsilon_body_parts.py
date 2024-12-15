import ast
import csv

from tqdm import tqdm

from epsdatasets.helpers.gradient.gradient_body_parts import MONAI_TO_EPSILON_BODY_PARTS_MAPPING
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils

import config


def get_anatomical_plane(file_name):
    try:
        txt_file = file_name.replace(".nii.gz", ".txt")
        content =  gcs_utils.download_file_as_string(config.GCS_BUCKET_NAME, txt_file)
        lines = content.split('\n')

        image_orientation = None
        dicom_tag = "(0020,0037) Image Orientation (Patient): "

        for line in lines:
            start_index = line.find(dicom_tag)
            if start_index != -1:
                start_index += len(dicom_tag)
                image_orientation = ast.literal_eval(line[start_index:].strip())

        if image_orientation is None:
            return None

        plane = dicom_utils.get_anatomical_plane(image_orientation)
        return plane

    except Exception as e:
        print(f"Error getting anatomical plane from {txt_file}: {str(e)}")
        return None

def main():
    print(f"Reading input file {config.CHECK_BODY_PARTS_OUTPUT_FILE}")

    out_data = []
    with open(config.CHECK_BODY_PARTS_OUTPUT_FILE, "r") as file:
        for line in tqdm(file, desc="Processing"):
            if line.startswith("ERROR:"):
                continue

            labels, file_name = line.strip().rsplit("},", 1)
            labels += "}"
            labels = ast.literal_eval(labels)

            # Reject non-axial CTs?
            if config.REJECT_NON_AXIAL:
                plane = get_anatomical_plane(file_name)
                if plane != dicom_utils.AnatomicalPlane.AXIAL:
                    continue

            if labels["top_label"] is None:
                continue

            top_label = MONAI_TO_EPSILON_BODY_PARTS_MAPPING[labels["top_label"]]
            out_data.append([file_name, top_label])

    print(f"Writing to output file {config.CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE}")

    with open(config.CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Volume", "BodyPart"])
        writer.writerows(out_data)


if __name__ == "__main__":
    main()
