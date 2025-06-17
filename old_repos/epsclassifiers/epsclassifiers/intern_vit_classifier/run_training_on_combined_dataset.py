import argparse
import torch
import yaml

from epsclassifiers.intern_vit_classifier import InternVitClassifier
from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


def convert_none(value):
    if value == "None":
        return None

    return value


def main(config_path):
    # Read the configuration file.
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get the configuration parameters.
    model_name                     = config["general"].get("model_name", "")
    dataset_name                   = config["general"].get("dataset_name", "")
    run_name                       = config["general"].get("run_name", "")
    notes                          = config["general"].get("notes", "")
    custom_labels                  = convert_none(config["general"].get("custom_labels", None))
    body_part                      = config["general"].get("body_part", "")
    treat_uncertain_as_positive    = config["general"].get("treat_uncertain_as_positive", False)
    save_full_model                = config["general"].get("save_full_model", False)
    intern_vl_checkpoint_dir       = config["paths"].get("intern_vl_checkpoint_dir", "")
    train_file                     = config["paths"].get("train_file", "")
    validation_file                = config["paths"].get("validation_file", "")
    test_file                      = config["paths"].get("test_file", "")
    base_path_substitutions        = convert_none(config["paths"].get("base_path_substitutions", None))
    output_dir                     = config["paths"].get("output_dir", "")
    perform_intra_epoch_validation = config["training"].get("perform_intra_epoch_validation", False)
    intra_epoch_validation_step    = config["training"].get("intra_epoch_validation_step", 5000)
    send_wandb_notification        = config["training"].get("send_wandb_notification", True)
    device                         = config["training"].get("device", "")
    device_ids                     = convert_none(config["training"].get("device_ids", None))
    num_training_workers_per_gpu   = config["training"].get("num_training_workers_per_gpu", 1)
    num_validation_workers_per_gpu = config["training"].get("num_validation_workers_per_gpu", 1)
    learning_rate                  = config["training"].get("learning_rate", 1e-6)
    warmup_ratio                   = config["training"].get("warmup_ratio", 0.1)
    num_epochs                     = config["training"].get("num_epochs", 1)
    training_batch_size            = config["training"].get("training_batch_size", 1)
    validation_batch_size          = config["training"].get("validation_batch_size", 1)
    min_allowed_batch_size         = config["training"].get("min_allowed_batch_size", 1)
    multi_image_input              = config["training"].get("multi_image_input", False)
    num_multi_images               = convert_none(config["training"].get("num_multi_images", None))

    # Print configuration parameters.
    print("----------------------------------------------------------")
    print("Using the following configuration parameters:")
    print(f"+ model_name: {model_name}")
    print(f"+ dataset_name: {dataset_name}")
    print(f"+ run_name: {run_name}")
    print(f"+ notes: {notes}")
    print(f"+ custom_labels: {custom_labels}")
    print(f"+ body_part: {body_part}")
    print(f"+ treat_uncertain_as_positive: {treat_uncertain_as_positive}")
    print(f"+ save_full_model: {save_full_model}")
    print(f"+ intern_vl_checkpoint_dir: {intern_vl_checkpoint_dir}")
    print(f"+ train_file: {train_file}")
    print(f"+ validation_file: {validation_file}")
    print(f"+ test_file: {test_file}")
    print(f"+ base_path_substitutions: {base_path_substitutions}")
    print(f"+ output_dir: {output_dir}")
    print(f"+ perform_intra_epoch_validation: {perform_intra_epoch_validation}")
    print(f"+ intra_epoch_validation_step: {intra_epoch_validation_step}")
    print(f"+ send_wandb_notification: {send_wandb_notification}")
    print(f"+ device: {device}")
    print(f"+ device_ids: {device_ids}")
    print(f"+ num_training_workers_per_gpu: {num_training_workers_per_gpu}")
    print(f"+ num_validation_workers_per_gpu: {num_validation_workers_per_gpu}")
    print(f"+ learning_rate: {learning_rate}")
    print(f"+ warmup_ratio: {warmup_ratio}")
    print(f"+ num_epochs: {num_epochs}")
    print(f"+ training_batch_size: {training_batch_size}")
    print(f"+ validation_batch_size: {validation_batch_size}")
    print(f"+ min_allowed_batch_size: {min_allowed_batch_size}")
    print(f"+ multi_image_input: {multi_image_input}")
    print(f"+ num_multi_images: {num_multi_images}")
    print("----------------------------------------------------------")

    # Auto-generated names. Don't change.
    experiment_name = f"{model_name}-training-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GenericDatasetHelper(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        base_path_substitutions=base_path_substitutions,
        body_part=body_part,
        merge_val_and_test=True,
        treat_uncertain_as_positive=treat_uncertain_as_positive,
        convert_images_to_rgb=True,
        custom_labels=custom_labels)

    print(f"Using the following labels: {dataset_helper.get_labels()}")

    # Create the model.
    print("Creating the model")

    model = InternVitClassifier(num_classes=len(dataset_helper.get_labels()),
                                intern_vl_checkpoint_dir=intern_vl_checkpoint_dir,
                                intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                                multi_image_input=multi_image_input,
                                num_multi_images=num_multi_images,
                                use_text_encodings=False)

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
        if multi_image_input:
            try:
                stack = []
                for item in samples:
                    images = dataset_helper.get_pil_image(item)
                    assert len(images) == num_multi_images
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    stack.append(pixel_values)

                pixel_values = torch.stack(stack)
                pixel_values = pixel_values.to(torch.bfloat16)
                return pixel_values
            except:
                return None

        else:
            try:
                images = [dataset_helper.get_pil_image(item)[0] for item in samples]
                pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(torch.bfloat16)
                return pixel_values
            except:
                return None

    def get_torch_labels(samples):
        labels = torch.stack([dataset_helper.get_torch_label(item).to(torch.bfloat16) for item in samples])
        return labels

    def collate_function(samples):
        images = get_torch_images(samples)

        if images is None:
            return None

        labels = get_torch_labels(samples)

        data = {
            "images": images,
            "file_names": [sample["image_path"] for sample in samples]
        }

        return data, labels

    training_helper.start_training(collate_function_for_training=collate_function)

    if save_full_model:
        training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternViT classifier training script")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    main(args.config_path)
