import argparse
import json
import torch
import torch.multiprocessing

from transformers import DistilBertTokenizer

from epsclassifiers.intern_vit_classifier import InternVitClassifier
from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.config.config_loader import ConfigLoader
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters


def main(config_path):
    # Set the sharing strategy to file system.
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Read the configuration file.
    config = ConfigLoader().load_config(config_path)
    config_str = json.dumps(config, indent=4)

    # Print configuration parameters.
    print("----------------------------------------------------------")
    print("Using the following configuration parameters:")
    print(config_str)
    print("----------------------------------------------------------")

    # Auto-generated names. Don't change.
    experiment_dir = f"{config['paths']['output_dir']}/{config['general']['experiment_name']}"
    save_model_filename = f"{experiment_dir}/{config['general']['experiment_name']}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{config['general']['experiment_name']}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GenericDatasetHelper(
        train_file=config["paths"]["train_file"],
        validation_file=config["paths"]["validation_file"],
        test_file=config["paths"]["test_file"],
        base_path_substitutions=config["paths"]["base_path_substitutions"],
        body_part=config["data"]["body_part"],
        sub_body_part=config["data"]["sub_body_part"],
        merge_val_and_test=True,
        treat_uncertain_as_positive=config["data"]["treat_uncertain_as_positive"],
        perform_label_balancing=config["data"]["perform_label_balancing"],
        negative_body_parts_ratio=config["data"]["negative_body_parts_ratio"],
        num_data_augmentations=config["data"]["num_data_augmentations"],
        compute_num_data_augmentations=config["data"]["compute_num_data_augmentations"],
        data_augmentation_target=config["data"]["data_augmentation_target"],
        data_augmentation_min=config["data"]["data_augmentation_min"],
        max_study_images=config["data"]["max_study_images"],
        convert_images_to_rgb=True,
        replace_dicom_with_png=config["data"]["replace_dicom_with_png"],
        custom_labels=config["data"]["custom_labels"])

    print(f"Using the following labels: {dataset_helper.get_labels()}")

    # Create the tokenizer.
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Create the model.
    print("Creating the model")

    model = InternVitClassifier(num_classes=len(dataset_helper.get_labels()),
                                intern_vl_checkpoint_dir=config["paths"]["backbone_checkpoint_dir"],
                                intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                                multi_image_input=config["training"]["multi_image_input"],
                                num_multi_images=config["training"]["num_multi_images"],
                                use_text_encodings=config["training"]["use_report_text"],
                                use_attentional_pooling=config["training"]["use_attentional_pooling"])

    model = model.to("cuda")
    image_processor = model.get_image_processor()

    for param in model.parameters():
        param.requires_grad = True

    # Freeze the InternViT.
    for param in model.intern_vit.parameters():
        param.requires_grad = False

    # Prepare the training data.
    print("Preparing the training data")

    training_parameters = TrainingParameters(learning_rate=config["training"]["learning_rate"],
                                             warmup_ratio=config["training"]["warmup_ratio"],
                                             num_epochs=config["training"]["num_epochs"],
                                             training_batch_size=config["training"]["training_batch_size"],
                                             validation_batch_size=config["training"]["validation_batch_size"],
                                             min_allowed_batch_size=config["training"]["min_allowed_batch_size"],
                                             criterion=SampleBalancedBCEWithLogitsLoss(),
                                             checkpoint_dir=checkpoint_dir,
                                             perform_intra_epoch_validation=config["training"]["perform_intra_epoch_validation"],
                                             intra_epoch_validation_step=config["training"]["intra_epoch_validation_step"],
                                             num_training_workers_per_gpu=config["training"]["num_training_workers_per_gpu"],
                                             num_validation_workers_per_gpu=config["training"]["num_validation_workers_per_gpu"],
                                             save_visualizaton_data_during_training=True,
                                             save_visualizaton_data_during_validation=True,
                                             pause_on_validation_visualization=False)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=config["general"]["experiment_name"],
                                       run_name=config["general"]["run_name"],
                                       notes=config["general"]["notes"],
                                       label_names=dataset_helper.get_labels(),
                                       send_notification=config["training"]["send_wandb_notification"])

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=config["training"]["device"],
                                          device_ids=config["training"]["device_ids"],
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters,
                                          multi_gpu_padding=(config["training"]["multi_image_input"] and config["training"]["num_multi_images"] is None),
                                          config_file_content=config_str)

    def get_torch_images(samples):
        if config["training"]["multi_image_input"] and config["training"]["num_multi_images"] is not None:
            try:
                stack = []
                for item in samples:
                    images = dataset_helper.get_pil_image(item)
                    assert len(images) == config["training"]["num_multi_images"]
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    stack.append(pixel_values)

                pixel_values = torch.stack(stack)
                pixel_values = pixel_values.to(torch.bfloat16)
                return pixel_values
            except:
                return None

        elif config["training"]["multi_image_input"] and config["training"]["num_multi_images"] is None:
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

        report_texts, text_encodings = get_text_encodings(samples) if config["training"]["use_report_text"] else (None, None)
        labels = get_torch_labels(samples)

        data = {
            "images": images,
            "report_texts": report_texts,
            "text_encodings": text_encodings,
            "file_names": [sample["relative_image_paths"] for sample in samples],
        }

        return data, labels

    training_helper.start_training(collate_function_for_training=collate_function)

    if config["general"]["save_full_model"]:
        training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternViT classifier training script")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    main(args.config_path)
