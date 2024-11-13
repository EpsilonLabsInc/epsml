import numpy as np
import torch
import torchvision

from epsdatasets.helpers.gradient.gradient_dataset_helper import GradientDatasetHelper
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


if __name__ == "__main__":
    # General settings.
    model_name = "x3d"
    dataset_name = "ct_chest_training_sample_reduced"

    # Gradient dataset helper settings.
    images_dir = "16AGO2024"
    reports_file = None
    grouped_labels_file = "/home/andrej/data/gradient/grouped_labels_GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1.json"
    images_index_file = None
    generated_data_file = "/home/andrej/data/gradient/ct_chest_training_sample_reduced.csv"
    output_dir = "/home/andrej/data/gradient/output"
    perform_quality_check = False
    gcs_bucket_name = "gradient-cts-nifti"
    modality = "CT"
    min_volume_depth = 10  # Skip volumes with < 10 slices.
    max_volume_depth = 200  # Skip volumes with > 200 slices.
    preserve_image_format = True
    seed = 42
    run_statistics = False

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = True
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    num_training_workers_per_gpu = 8
    num_validation_workers_per_gpu = 8
    half_model_precision = False
    learning_rate = 1e-5
    warmup_ratio = 1 / 6
    num_epochs = 3
    num_steps_per_checkpoint = 5000
    gradient_accumulation_steps = 4
    training_batch_size = 2
    validation_batch_size = 2
    images_mean = 0.2567
    images_std = 0.1840
    target_image_size = 224
    normalization_depth = 112
    loss_function = SampleBalancedBCEWithLogitsLoss()  # torch.nn.BCEWithLogitsLoss()

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_weights_filename = f"{experiment_dir}/{experiment_name}-weights.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GradientDatasetHelper(
        display_name=dataset_name,
        images_dir=images_dir,
        reports_file=reports_file,
        grouped_labels_file=grouped_labels_file,
        images_index_file=images_index_file,
        generated_data_file=generated_data_file,
        output_dir=output_dir,
        perform_quality_check=perform_quality_check,
        gcs_bucket_name=gcs_bucket_name,
        modality=modality,
        min_volume_depth=min_volume_depth,
        max_volume_depth=max_volume_depth,
        preserve_image_format=preserve_image_format,
        use_half_precision=half_model_precision,
        seed=seed,
        run_statistics=run_statistics)

    print(f"Target volume dimensions: {target_image_size}x{target_image_size}x{normalization_depth}")

    # Get number of labels.
    num_labels = len(dataset_helper.get_labels())
    print(f"Number of labels: {num_labels}")

    # Create the model and replace its input and head layers.
    print("Creating the X3D model")
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
    model.blocks[0].conv.conv_t = torch.nn.Conv3d(1, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
    model.blocks[5].proj = torch.nn.Linear(model.blocks[5].proj.in_features, num_labels)

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
                                             criterion=loss_function,
                                             checkpoint_dir=checkpoint_dir,
                                             perform_intra_epoch_validation=perform_intra_epoch_validation,
                                             num_steps_per_checkpoint = num_steps_per_checkpoint,
                                             num_training_workers_per_gpu=num_training_workers_per_gpu,
                                             num_validation_workers_per_gpu=num_validation_workers_per_gpu)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=mlops_experiment_name,
                                       notes=f"Volume size = {target_image_size}x{target_image_size}x{normalization_depth}",
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    transform_rgb_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize((target_image_size, target_image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
    ])

    def transform_uint16_image(image):
        image = image.resize((target_image_size, target_image_size))
        image_np = np.array(image).astype(np.float32)
        image_np /= 65535.0
        image_np = image_np.astype(np.float16) if half_model_precision else image_np.astype(np.float32)
        image_np = (image_np - images_mean) / images_std
        image_tensor = torch.from_numpy(image_np)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def get_torch_image(item):
        images = dataset_helper.get_pil_image(item, normalization_depth, sample_slices=True)
        tensors = [transform_uint16_image(image) for image in images]
        stacked_tensor = torch.stack(tensors)
        # Instead of the tensor shape (num_slices, num_channels, image_height, image_width),
        # which is obtained by stacking the tensors, the model requires the following shape:
        # (num_channels, num_slices, image_height, image_width), which is obtained by
        # premuting the dimensions.
        stacked_tensor = stacked_tensor.permute(1, 0, 2, 3)
        return stacked_tensor

    def collate_function(samples):
        images = torch.stack([get_torch_image(item=item) for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    training_helper.start_training(collate_function=collate_function)
    training_helper.save_model_weights(model_weights_file_name=save_model_weights_filename)
