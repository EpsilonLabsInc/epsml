import torch
import torchxrayvision as xrv

from epsdatasets.helpers.gradient_mimic.gradient_mimic_dataset_helper import GradientMimicDatasetHelper
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


if __name__ == "__main__":
    # General settings.
    model_name = "cr_body_part_classifier"
    dataset_name = "gradient_mimic"
    output_dir = "./output"

    # Gradient-Mimic dataset helper.
    gradient_data_gcs_bucket_name="gradient-crs"
    gradient_data_gcs_dir="16AG02924"
    gradient_images_gcs_bucket_name="epsilon-data-us-central1"
    gradient_images_gcs_dir="GRADIENT-DATABASE/CR/16AG02924"
    mimic_gcs_bucket_name="epsilonlabs-filestore"
    mimic_gcs_dir="mimic2-dicom/mimic-cxr-jpg-2.1.0.physionet.org"
    exclude_file_name="/home/andrej/work/epsclassifiers/epsclassifiers/cr_body_part_classifier/chest_scan_results.txt"
    seed=42

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = True
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    num_training_workers_per_gpu = 8
    num_validation_workers_per_gpu = 8
    half_model_precision = False
    learning_rate = 1e-3
    warmup_ratio = 1 / 10
    num_epochs = 30
    training_batch_size = 16
    validation_batch_size = 16
    images_mean = 0.5
    images_std = 0.5

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GradientMimicDatasetHelper(gradient_data_gcs_bucket_name=gradient_data_gcs_bucket_name,
                                                gradient_data_gcs_dir=gradient_data_gcs_dir,
                                                gradient_images_gcs_bucket_name=gradient_images_gcs_bucket_name,
                                                gradient_images_gcs_dir=gradient_images_gcs_dir,
                                                mimic_gcs_bucket_name=mimic_gcs_bucket_name,
                                                mimic_gcs_dir=mimic_gcs_dir,
                                                exclude_file_name=exclude_file_name,
                                                seed=42)

    # Create the model.
    model = xrv.models.DenseNet(num_classes=1)

    for param in model.parameters():
        param.requires_grad = True

    if half_model_precision:
        model.half()

    # Prepare the training data.
    print("Preparing the training data")

    training_parameters = TrainingParameters(learning_rate=learning_rate,
                                             warmup_ratio=warmup_ratio,
                                             num_epochs=num_epochs,
                                             training_batch_size=training_batch_size,
                                             validation_batch_size=validation_batch_size,
                                             criterion=torch.nn.BCEWithLogitsLoss(),
                                             checkpoint_dir=checkpoint_dir,
                                             perform_intra_epoch_validation=perform_intra_epoch_validation,
                                             num_training_workers_per_gpu=num_training_workers_per_gpu,
                                             num_validation_workers_per_gpu=num_validation_workers_per_gpu)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=mlops_experiment_name,
                                       notes="600K samples",
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    training_helper.start_training()
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
