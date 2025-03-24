import torch

from epsclassifiers.intern_vit_classifier import InternVitClassifier
from epsdatasets.helpers.mimic.v2.mimic_two_dataset_helper import MimicTwoDatasetHelper
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


def main():
    # General settings.
    model_name = "intern_vit_classifier"
    dataset_name = "mimic_two_cardiomegaly"
    binary_label = "Cardiomegaly"
    run_name = "26B with no labels"
    notes = "InternVL model: 26B with no labels"
    save_full_model = False

    # Paths.
    intern_vl_checkpoint_dir = "/mnt/efs/models/internvl/old/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
    dataset_gcs_uri = "gs://epsilonlabs-filestore/mimic2-dicom/mimic-cxr-jpg-2.1.0.physionet.org"
    output_dir = "./output"

    # Training settings.
    perform_intra_epoch_validation = True
    intra_epoch_validation_step = 1000
    send_wandb_notification = False
    device = "cuda"
    device_ids = None
    num_training_workers_per_gpu = 8
    num_validation_workers_per_gpu = 8
    learning_rate = 0.01
    warmup_ratio = 0.05
    num_epochs = 4
    training_batch_size = 16
    validation_batch_size = 16
    min_allowed_batch_size = 2  # In order for batch norm in the InternVitClassifier model to work.
    multi_image_input = False
    num_multi_images = None

    # Auto-generated names. Don't change.
    experiment_name = f"{model_name}-training-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = MimicTwoDatasetHelper(gcs_uri=dataset_gcs_uri, binary_label=binary_label)

    print(f"Using the following labels: {dataset_helper.get_labels()}")

    # Create the model.
    print("Creating the model")
    model = InternVitClassifier(num_classes=len(dataset_helper.get_labels()),
                                intern_vl_checkpoint_dir=intern_vl_checkpoint_dir,
                                intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                                multi_image_input=multi_image_input,
                                num_multi_images=num_multi_images)
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
                                             min_allowed_batch_size=min_allowed_batch_size,
                                             criterion=SampleBalancedBCEWithLogitsLoss(),
                                             checkpoint_dir=checkpoint_dir,
                                             perform_intra_epoch_validation=perform_intra_epoch_validation,
                                             intra_epoch_validation_step=intra_epoch_validation_step,
                                             num_training_workers_per_gpu=num_training_workers_per_gpu,
                                             num_validation_workers_per_gpu=num_validation_workers_per_gpu,
                                             save_visualizaton_data_during_training=True,
                                             save_visualizaton_data_during_validation=True,
                                             pause_on_validation_visualization=False)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=mlops_experiment_name,
                                       run_name=run_name,
                                       notes=notes,
                                       label_names=dataset_helper.get_labels(),
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
        return images, labels, [sample["image_path"] for sample in samples]

    training_helper.start_training(collate_function_for_training=collate_function)

    if save_full_model:
        training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)


if __name__ == "__main__":
    main()
