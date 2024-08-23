import copy
import sys

sys.path.insert(1, "/home/ec2-user/work/registry/dicom_converter")
sys.path.insert(1, "/home/ec2-user/work/registry/helpers")
sys.path.insert(1, "/home/ec2-user/work/registry/mimic")

import torch
import torchvision

from i3d_resnet import I3DResNet
from fawkes_dataset_helper import FawkesDatasetHelper
from training_helper import TrainingHelper, TrainingParameters, MlFlowParameters


if __name__ == "__main__":
    model_name = "inflated_resnet"
    dataset_name = "fawkes_varying_dataset"
    labeled_data_file="/home/ec2-user/data/mnt/epsilon-datasets/fawkes/fawkes_varying_volumes/labeled_data.json"
    grouped_labels_file="/home/ec2-user/data/mnt/epsilon-datasets/fawkes/grouped_labels.json"
    mlflow_uri = "https://mlflow-f66025e-rcsxwgoiba-uc.a.run.app"

    device = "cuda"
    # device_ids = None  # Use one (the default) GPU.
    device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    half_model_precision = False
    num_epochs = 10
    batch_size = 1
    seed = 42

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlflow_experiment_name = f"{experiment_name}"
    out_dir = f"/home/ec2-user/data/{experiment_name}"
    save_model_filename = f"{out_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{out_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{out_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = FawkesDatasetHelper(
        labeled_data_file=labeled_data_file, grouped_labels_file=grouped_labels_file, use_half_precision=half_model_precision, seed=seed)
    max_depth = dataset_helper.get_max_depth()

    # Volume depth must be greater than max_depth and divisible by 8 (the latter is I3D's constraint).
    volume_depth = ((max_depth // 8) + 1) * 8
    print(f"Max depth in the dataset is {max_depth}, setting volume depth to {volume_depth}")

    # Get labels manager.
    labels_manager = dataset_helper.get_labels_manager()
    num_labels = len(labels_manager.get_groups())

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
    training_parameters = TrainingParameters(num_epochs=num_epochs, batch_size=batch_size, criterion=torch.nn.BCEWithLogitsLoss(), checkpoint_dir=checkpoint_dir)
    mlflow_parameters = MlFlowParameters(uri=mlflow_uri, experiment_name=mlflow_experiment_name)
    training_helper = TrainingHelper(model=model,
                                    dataset_helper=dataset_helper,
                                    device=device,
                                    device_ids=device_ids,
                                    training_parameters=training_parameters,
                                    mlflow_parameters=mlflow_parameters)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
        # TODO: Figure out normalization values. Also consider computing mean and std from the dataset itself.
#        torchvision.transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25])
    ])

    def collate_function(samples):
        images = torch.stack([dataset_helper.get_torch_image(item=item, transform=transform, normalization_depth=volume_depth) for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    training_helper.start_torch_training(collate_function=collate_function)
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
