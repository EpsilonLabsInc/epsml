import argparse
import torch
import torch.multiprocessing
import yaml

from transformers import DistilBertTokenizer

from epsclassifiers.intern_vit_classifier import InternVitClassifier
from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


def convert_none(value):
    if value == "None":
        return None

    return value


def main(config_path):
    # Set the sharing strategy to file system.
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Read the configuration file.
    with open(config_path, "r") as file:
        config_file_content = file.read()

    # Load config.
    config = yaml.safe_load(config_file_content)

    # Get the configuration parameters.
    experiment_name                = config["general"].get("experiment_name", "")
    run_name                       = config["general"].get("run_name", "")
    notes                          = config["general"].get("notes", "")
    save_full_model                = config["general"].get("save_full_model", False)
    intern_vl_checkpoint_dir       = config["paths"].get("intern_vl_checkpoint_dir", "")
    train_file                     = config["paths"].get("train_file", "")
    validation_file                = config["paths"].get("validation_file", "")
    test_file                      = config["paths"].get("test_file", "")
    base_path_substitutions        = convert_none(config["paths"].get("base_path_substitutions", None))
    output_dir                     = config["paths"].get("output_dir", "")
    body_part                      = config["data"].get("body_part", "")
    sub_body_part                  = convert_none(config["data"].get("sub_body_part", None))
    custom_labels                  = convert_none(config["data"].get("custom_labels", None))
    treat_uncertain_as_positive    = config["data"].get("treat_uncertain_as_positive", False)
    perform_label_balancing        = config["data"].get("perform_label_balancing", False)
    num_data_augmentations         = config["data"].get("num_data_augmentations", 0)
    compute_num_data_augmentations = config["data"].get("compute_num_data_augmentations", False)
    data_augmentation_target       = config["data"].get("data_augmentation_target", 0)
    data_augmentation_min          = config["data"].get("data_augmentation_min", 0)
    max_study_images               = convert_none(config["data"].get("max_study_images", None))
    replace_dicom_with_png         = config["data"].get("replace_dicom_with_png", False)
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
    use_report_text                = config["training"].get("use_report_text", False)
    multi_image_input              = config["training"].get("multi_image_input", False)
    num_multi_images               = convert_none(config["training"].get("num_multi_images", None))
    use_attentional_pooling        = config["training"].get("use_attentional_pooling", False)

    # Print configuration parameters.
    print("----------------------------------------------------------")
    print("Using the following configuration parameters:")
    print(f"+ experiment_name: {experiment_name}")
    print(f"+ run_name: {run_name}")
    print(f"+ notes: {notes}")
    print(f"+ save_full_model: {save_full_model}")
    print(f"+ intern_vl_checkpoint_dir: {intern_vl_checkpoint_dir}")
    print(f"+ train_file: {train_file}")
    print(f"+ validation_file: {validation_file}")
    print(f"+ test_file: {test_file}")
    print(f"+ base_path_substitutions: {base_path_substitutions}")
    print(f"+ output_dir: {output_dir}")
    print(f"+ body_part: {body_part}")
    print(f"+ sub_body_part: {sub_body_part}")
    print(f"+ custom_labels: {custom_labels}")
    print(f"+ treat_uncertain_as_positive: {treat_uncertain_as_positive}")
    print(f"+ perform_label_balancing: {perform_label_balancing}")
    print(f"+ num_data_augmentations: {num_data_augmentations}")
    print(f"+ compute_num_data_augmentations: {compute_num_data_augmentations}")
    print(f"+ data_augmentation_target: {data_augmentation_target}")
    print(f"+ data_augmentation_min: {data_augmentation_min}")
    print(f"+ max_study_images: {max_study_images}")
    print(f"+ replace_dicom_with_png: {replace_dicom_with_png}")
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
    print(f"+ use_report_text: {use_report_text}")
    print(f"+ multi_image_input: {multi_image_input}")
    print(f"+ num_multi_images: {num_multi_images}")
    print(f"+ use_attentional_pooling: {use_attentional_pooling}")
    print("----------------------------------------------------------")

    # Auto-generated names. Don't change.
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
        sub_body_part=sub_body_part,
        merge_val_and_test=True,
        treat_uncertain_as_positive=treat_uncertain_as_positive,
        perform_label_balancing=perform_label_balancing,
        num_data_augmentations=num_data_augmentations,
        compute_num_data_augmentations=compute_num_data_augmentations,
        data_augmentation_target=data_augmentation_target,
        data_augmentation_min=data_augmentation_min,
        max_study_images=max_study_images,
        convert_images_to_rgb=True,
        replace_dicom_with_png=replace_dicom_with_png,
        custom_labels=custom_labels)

    print(f"Using the following labels: {dataset_helper.get_labels()}")

    # Create the tokenizer.
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Create the model.
    print("Creating the model")

    model = InternVitClassifier(num_classes=len(dataset_helper.get_labels()),
                                intern_vl_checkpoint_dir=intern_vl_checkpoint_dir,
                                intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                                multi_image_input=multi_image_input,
                                num_multi_images=num_multi_images,
                                use_text_encodings=use_report_text,
                                use_attentional_pooling=use_attentional_pooling)

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
                                       experiment_name=experiment_name,
                                       run_name=run_name,
                                       notes=notes,
                                       label_names=dataset_helper.get_labels(),
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters,
                                          multi_gpu_padding=(multi_image_input and num_multi_images is None),
                                          config_file_content=config_file_content)

    device_ids_used = training_helper.get_device_ids_used()

    def get_torch_images(samples):
        if multi_image_input and num_multi_images is not None:
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

        elif multi_image_input and num_multi_images is None:
            try:
                image_list = []
                for item in samples:
                    images = dataset_helper.get_pil_image(item)
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(torch.bfloat16)
                    image_list.append(pixel_values)
                return image_list
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

    def get_text_encodings(samples):
        report_texts = [dataset_helper.get_report_text(item) for item in samples]
        text_encodings = tokenizer(report_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return report_texts, text_encodings

    def get_torch_labels(samples):
        labels = torch.stack([dataset_helper.get_torch_label(item).to(torch.bfloat16) for item in samples])
        return labels

    def collate_function(samples):
        images = get_torch_images(samples)

        if images is None:
            return None

        report_texts, text_encodings = get_text_encodings(samples) if use_report_text else (None, None)
        labels = get_torch_labels(samples)

        data = {
            "images": images,
            "report_texts": report_texts,
            "text_encodings": text_encodings,
            "file_names": [sample["image_paths"] for sample in samples],
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
