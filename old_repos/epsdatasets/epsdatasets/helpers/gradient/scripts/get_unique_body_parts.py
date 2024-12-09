from epsdatasets.helpers.gradient import gradient_utils

# GCS_BUCKET_NAME = "gradient-cts-nifti"
# REPORTS_FILE_PATH = "to_be_deleted/kedar-body-part/GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1_bodyPart_contrast_anon_check_filtered_us_data.csv"

GCS_BUCKET_NAME = None
REPORTS_FILE_PATH = "C:/Users/Andrej/Desktop/to_be_deleted_kedar-body-part_GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1_bodyPart_contrast_anon_check_filtered_us_data.csv"


def main():
    all_body_parts = gradient_utils.get_all_body_parts_from_report(reports_file_path=REPORTS_FILE_PATH, gcs_bucket_name=GCS_BUCKET_NAME)
    all_body_parts = [item for sublist in all_body_parts for item in sublist]  # Flatten list.
    print(f"All body parts found: {len(all_body_parts)}")

    unique_body_parts = gradient_utils.get_unique_body_parts_from_report(reports_file_path=REPORTS_FILE_PATH, gcs_bucket_name=GCS_BUCKET_NAME)
    print(f"All unique body parts found: {len(unique_body_parts)}")

    distribution = {}
    for body_part in all_body_parts:
        if body_part in distribution:
            distribution[body_part] += 1
        else:
            distribution[body_part] = 1

    print(f"Body part distribution:")
    print(distribution)


if __name__ == "__main__":
    main()
