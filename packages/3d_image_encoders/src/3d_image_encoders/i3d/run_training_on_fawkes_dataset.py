import copy

import torch
import torchvision

from epsdatasets.helpers.fawkes.fawkes_dataset_helper import FawkesDatasetHelper
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters

from i3d_resnet import I3DResNet


if __name__ == "__main__":
    model_name = "inflated_resnet"
    dataset_name = "fawkes_varying_dataset"
    labeled_data_file="/home/ec2-user/data/mnt/epsilon-datasets/fawkes/fawkes_varying_volumes/labeled_data.json"
    grouped_labels_file="/home/ec2-user/data/mnt/epsilon-datasets/fawkes/grouped_labels.json"

    device = "cuda"
    # device_ids = None  # Use one (the default) GPU.
    device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    volume_depth_threshold = 200  # Skip volumes with >= 200 slices.
    half_model_precision = False
    learning_rate = 1e-6
    num_epochs = 10
    training_batch_size = 4
    validation_batch_size = 4
    seed = 42

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    out_dir = f"/home/ec2-user/data/{experiment_name}"
    save_model_filename = f"{out_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{out_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{out_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = FawkesDatasetHelper(
        labeled_data_file=labeled_data_file, grouped_labels_file=grouped_labels_file,
        volume_depth_threshold=volume_depth_threshold, use_half_precision=half_model_precision,
        seed=seed)
    max_depth = dataset_helper.get_max_depth()

    # Volume depth must be greater than max_depth and divisible by 8 (the latter is I3D's constraint).
    volume_depth = ((max_depth // 8) + 1) * 8
    print(f"Max depth in the dataset is {max_depth}, setting volume depth to {volume_depth}")

    # Get number of labels.
    num_labels = len(dataset_helper.get_labels())
    print(f"Number of labels: {num_labels}")

    # Create the model.
    print("Creating the model")
    resnet = torchvision.models.resnet152(pretrained=True)
    model = I3DResNet(resnet2d=copy.deepcopy(resnet), frame_nb=volume_depth, class_nb=num_labels, conv_class=True)

    for param in model.parameters():
        param.requires_grad = True

    if half_model_precision:
        model.half()

    # Prepare the training data.
    print("Preparing the training data")

    training_parameters = TrainingParameters(learning_rate=learning_rate,
                                             num_epochs=num_epochs,
                                             training_batch_size=training_batch_size,
                                             validation_batch_size=validation_batch_size,
                                             criterion=torch.nn.BCEWithLogitsLoss(),
                                             checkpoint_dir=checkpoint_dir)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=mlops_experiment_name)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
    ])

    def collate_function(samples):
        images = torch.stack([dataset_helper.get_torch_image(item=item, transform=transform, normalization_depth=volume_depth) for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    training_helper.start_training(collate_function=collate_function)
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
