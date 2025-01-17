import torch

from epsdatasets.helpers.gradient_cr.gradient_cr_dataset_helper import GradientCrDatasetHelper


def main():
    gcs_train_file = "gs://epsilonlabs-filestore/cleaned_CRs/gradient_rm_bad_dcm_1211_nolabel.jsonl"
    gcs_validation_file = "gs://epsilonlabs-filestore/cleaned_CRs/11192024_test.jsonl"

    # TODO: Finish implementation.

    dataset_helper = GradientCrDatasetHelper(
        gcs_train_file=gcs_train_file,
        gcs_validation_file=gcs_validation_file
    )

    print(dataset_helper.get_labels())


if __name__ == "__main__":
    main()
