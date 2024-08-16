import copy
import sys

sys.path.insert(1, "/home/ec2-user/work/registry/helpers")
sys.path.insert(1, "/home/ec2-user/work/registry/mimic")

import torch
import torchvision

from i3d_resnet import I3DResNet
from covid_dataset_helper import CovidDatasetHelper
from training_helper import TrainingHelper, TrainingParameters, MlFlowParameters


if __name__ == "__main__":
    model_name = "inflated_resnet"
    dataset_name = "covid_dataset"
    dataset_path = "/home/ec2-user/data/New_Data_CoV2"

    # TODO: Remove.
    mlflow_user = "andrej"
    mlflow_pass = "ea`zrZT'V408"
    mlflow_uri = "https://mlflow-f66025e-rcsxwgoiba-uc.a.run.app"

    device = "cuda"
    device_ids = None
    num_epochs = 10
    batch_size = 5
    seed = 42

    # Required by ResNet.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlflow_experiment_name = f"{experiment_name}"
    out_dir = f"/home/ec2-user/data/{experiment_name}"
    save_model_filename = f"{out_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{out_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{out_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = CovidDatasetHelper(dataset_path=dataset_path, seed=seed)
    max_depth = dataset_helper.get_max_depth()

    # Volume depth must be greater than max_depth and divisible by 8 (the latter is I3D's constraint).
    volume_depth = ((max_depth // 8) + 1) * 8
    print(f"Max depth in the dataset is {max_depth}, setting volume depth to {volume_depth}")

    # Create the model.
    print("Creating the model")
    resnet = torchvision.models.resnet152(pretrained=True)
    model = I3DResNet(resnet2d=copy.deepcopy(resnet), frame_nb=volume_depth, class_nb=len(dataset_helper.get_labels()), conv_class=True)
    for param in model.parameters():
        param.requires_grad = True

    # Prepare the training data.
    print("Preparing the training data")
    training_parameters = TrainingParameters(num_epochs=num_epochs, batch_size=batch_size, checkpoint_dir=checkpoint_dir)
    mlflow_parameters = MlFlowParameters(username=mlflow_user, password=mlflow_pass, uri=mlflow_uri, experiment_name=mlflow_experiment_name)
    training_helper = TrainingHelper(model=model,
                                    dataset_helper=dataset_helper,
                                    device=device,
                                    device_ids=device_ids,
                                    training_parameters=training_parameters,
                                    mlflow_parameters=mlflow_parameters)

    def collate_function(samples):
        images = torch.stack([dataset_helper.get_torch_image(item=item, transform=transform, normalization_depth=volume_depth) for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    training_helper.start_torch_training(collate_function=collate_function)
    training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)
