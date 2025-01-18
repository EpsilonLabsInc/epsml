import torch

from epsclassifiers.intern_vit_classifier import InternVitClassifier
from epsdatasets.helpers.gradient_cr.gradient_cr_dataset_helper import GradientCrDatasetHelper
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


def main():
    # General settings.
    model_name = "intern_vit_classifier"
    dataset_name = "gradient_cr"
    output_dir = "./output"

    # Paths.
    intern_vl_checkpoint_dir = "/mnt/training/internvl2.5_8b_finetune_lora_20241226_205132_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6_no_labels/checkpoint-58670"
    gcs_train_file = "gs://epsilonlabs-filestore/cleaned_CRs/gradient_rm_bad_dcm_1211_nolabel.jsonl"
    gcs_validation_file = "gs://epsilonlabs-filestore/cleaned_CRs/11192024_test.jsonl"
    images_dir = "/mnt/gradient/gradient-cxr/22JUL2024"

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = True
    device = "cuda"
    # device_ids = None  # Use one (the default) GPU.
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # Use 8 GPUs.
    num_training_workers_per_gpu = 4
    num_validation_workers_per_gpu = 4
    learning_rate = 2e-4
    warmup_ratio = 1 / 10
    num_epochs = 4
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
    dataset_helper = GradientCrDatasetHelper(
        gcs_train_file=gcs_train_file,
        gcs_validation_file=gcs_validation_file,
        images_dir=images_dir
    )

    # Create the model.
    model = InternVitClassifier(num_classes=14, intern_vl_checkpoint_dir=intern_vl_checkpoint_dir)
    model = model.to("cuda")
    image_processor = model.get_image_processor()

    for param in model.parameters():
        param.requires_grad = True

    # Freeze the InternViT.
    for param in model.intern_vit.parameters():
        param.requires_grad = False

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
                                       notes="InternVL model: 8B with no labels",
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    def get_torch_images(samples):
        images = [dataset_helper.get_pil_image(item) for item in samples]
        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(torch.bfloat16)
        return pixel_values

    def get_torch_labels(samples):
        labels = torch.stack([dataset_helper.get_torch_label(item).to(torch.bfloat16) for item in samples])
        return labels

    def collate_function(samples):
        images = get_torch_images(samples)
        labels = get_torch_labels(samples)
        return images, labels

    training_helper.start_training(collate_function_for_training=collate_function)
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)


if __name__ == "__main__":
    main()
