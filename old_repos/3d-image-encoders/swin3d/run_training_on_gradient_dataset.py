import sys
sys.path.insert(1, "../../registry/mimic")
sys.path.insert(1, "../../registry/utils/dicom")
sys.path.insert(1, "../../registry/utils/labels")
sys.path.insert(1, "../../registry/utils/training")

import copy

import numpy as np
import torch
import torchvision

from custom_swin_3d import CustomSwin3D
from gradient_dataset_helper import GradientDatasetHelper
from torch_training_helper import TorchTrainingHelper, TrainingParameters, MlFlowParameters


if __name__ == "__main__":
    # General settings.
    model_name = "swin3d"
    dataset_name = "gradient-ct-16AGO2024"
    mlflow_uri = "https://mlflow-f66025e-rcsxwgoiba-uc.a.run.app"

    # Gradient dataset helper settings.
    images_dir = "data"
    reports_file = "/home/andrej/data/datasets/GRADIENT-DATABASE/REPORTS/CT/output_GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1.csv"
    grouped_labels_file = "/home/andrej/data/datasets/GRADIENT-DATABASE/REPORTS/CT/grouped_labels_GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1.json"
    images_index_file = None
    generated_data_file = "/home/andrej/data/datasets/GRADIENT-DATABASE/output/gradient-ct-16AGO2024-generated_data_nifti.csv"
    output_dir = "/home/andrej/data/datasets/GRADIENT-DATABASE/output"
    perform_quality_check = False
    gcs_bucket_name = "gradient-cts-nifti"
    modality = "CT"
    min_volume_depth = 10  # Skip volumes with < 10 slices.
    max_volume_depth = 200  # Skip volumes with > 200 slices.
    preserve_image_format = True
    seed = 42
    run_statistics = False

    # Training settings.
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    half_model_precision = True
    learning_rate = 1e-5
    num_epochs = 10
    batch_size = 1
    images_mean = 0.2567
    images_std = 0.1840

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlflow_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
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

    max_depth = dataset_helper.get_max_depth()

    # Volume depth must be greater than max_depth and divisible by 8 (the latter is I3D's constraint).
    volume_depth = ((max_depth // 8) + 1) * 8
    print(f"Max depth in the dataset is {max_depth}, setting volume depth to {volume_depth}")

    # Get number of labels.
    num_labels = len(dataset_helper.get_labels())
    print(f"Number of labels: {num_labels}")

    # Create the model.
    print("Creating the model")
    model = CustomSwin3D(
        model_size="tiny", num_classes=num_labels, use_pretrained_weights=True,
        use_single_channel_input=True, use_swin_v2=True, perform_gradient_checkpointing=True)

    for param in model.parameters():
        param.requires_grad = True

    if half_model_precision:
        model.half()

    # Prepare the training data.
    print("Preparing the training data")

    training_parameters = TrainingParameters(learning_rate=learning_rate,
                                             num_epochs=num_epochs,
                                             batch_size=batch_size,
                                             criterion=torch.nn.BCEWithLogitsLoss(),
                                             checkpoint_dir=checkpoint_dir)

    mlflow_parameters = MlFlowParameters(uri=mlflow_uri,
                                         experiment_name=mlflow_experiment_name)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlflow_parameters=mlflow_parameters)

    transform_rgb_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
        # TODO: Figure out normalization values. Also consider computing mean and std from the dataset itself.
#        torchvision.transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25])
    ])

    def transform_uint16_image(image):
        image = image.resize((512, 512))
        image_np = np.array(image).astype(np.float32)
        image_np /= 65535.0
        image_np = image_np.astype(np.float16) if half_model_precision else image_np.astype(np.float32)
        image_np = (image_np - images_mean) / images_std
        image_tensor = torch.from_numpy(image_np)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def get_torch_image(item):
        images = dataset_helper.get_pil_image(item, volume_depth)
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
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
