import torch

from epsclassifiers.cr_projection_classifier import CrProjectionClassifier
from epsdatasets.helpers.gradient_frontal_lateral.gradient_frontal_lateral_dataset_helper import GradientFrontalLateralDatasetHelper
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


if __name__ == "__main__":
    # General settings.
    model_name = "cr_projection_classifier"
    dataset_name = "gradient_frontal_lateral"
    notes = ""
    output_dir = "./output"

    # Gradient frontal/lateral dataset helper.
    gcs_chest_images_file = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
    gcs_frontal_images_file = "gs://gradient-crs/archive/projections/gradient-crs-22JUL2024-frontal-views.csv"
    gcs_lateral_images_file = "gs://gradient-crs/archive/projections/gradient-crs-22JUL2024-lateral-views.csv"
    gcs_bucket_name = "gs://epsilon-data-us-central1"
    seed = 42

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = False
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    num_training_workers_per_gpu = 32
    num_validation_workers_per_gpu = 32
    half_model_precision = False
    learning_rate = 1e-3
    warmup_ratio = 1 / 20
    num_epochs = 10
    training_batch_size = 32
    validation_batch_size = 32

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GradientFrontalLateralDatasetHelper(gcs_chest_images_file=gcs_chest_images_file,
                                                         gcs_frontal_images_file=gcs_frontal_images_file,
                                                         gcs_lateral_images_file=gcs_lateral_images_file,
                                                         gcs_bucket_name=gcs_bucket_name,
                                                         seed=seed):

    # Create the model.
    classifier = CrProjectionClassifier()
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
                                       notes=notes,
                                       label_names=dataset_helper.get_labels(),
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    training_helper.start_training()
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
