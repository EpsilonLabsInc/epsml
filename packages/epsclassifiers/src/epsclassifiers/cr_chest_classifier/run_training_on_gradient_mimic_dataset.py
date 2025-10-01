import torch

from epsclassifiers.cr_chest_classifier.cr_chest_classifier import CrChestClassifier
from epsdatasets.helpers.gradient_chest_non_chest.gradient_chest_non_chest_dataset_helper import GradientChestNonChestDatasetHelper
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


if __name__ == "__main__":
    # General settings.
    model_name = "cr_chest_classifier"
    dataset_name = "gradient_chest_non_chest"
    output_dir = "./output"

    # Gradient chest/non-chest dataset helper.
    chest_data_gcs_bucket_name = "gradient-crs"
    chest_data_gcs_dir = "22JUL2024"
    chest_images_gcs_bucket_name = "epsilon-data-us-central1"
    chest_images_gcs_dir = "GRADIENT-DATABASE/CR/22JUL2024"
    non_chest_data_gcs_bucket_name = "gradient-crs"
    non_chest_data_gcs_dir = "16AG02924"
    non_chest_images_gcs_bucket_name = "epsilon-data-us-central1"
    non_chest_images_gcs_dir = "GRADIENT-DATABASE/CR/16AG02924"
    chest_exclude_file_name = "./data/gradient_crs_22JUL2024_non_chest.csv"
    non_chest_exclude_file_name = "./data/gradient_crs_16AG02924_chest.csv"
    seed=42

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = False
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    num_training_workers_per_gpu = 16
    num_validation_workers_per_gpu = 16
    half_model_precision = False
    learning_rate = 1e-3
    warmup_ratio = 1 / 10
    num_epochs = 30
    training_batch_size = 16
    validation_batch_size = 16

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GradientChestNonChestDatasetHelper(chest_data_gcs_bucket_name=chest_data_gcs_bucket_name,
                                                        chest_data_gcs_dir=chest_data_gcs_dir,
                                                        chest_images_gcs_bucket_name=chest_images_gcs_bucket_name,
                                                        chest_images_gcs_dir=chest_images_gcs_dir,
                                                        non_chest_data_gcs_bucket_name=non_chest_data_gcs_bucket_name,
                                                        non_chest_data_gcs_dir=non_chest_data_gcs_dir,
                                                        non_chest_images_gcs_bucket_name=non_chest_images_gcs_bucket_name,
                                                        non_chest_images_gcs_dir=non_chest_images_gcs_dir,
                                                        chest_exclude_file_name=chest_exclude_file_name,
                                                        non_chest_exclude_file_name=non_chest_exclude_file_name,
                                                        seed=seed)

    # Create the model.
    classifier = CrChestClassifier()
    model = classifier.get_model()

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
                                             num_validation_workers_per_gpu=num_validation_workers_per_gpu,
                                             save_visualizaton_data_during_training=True,
                                             save_visualizaton_data_during_validation=True,
                                             pause_on_validation_visualization=False)

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
