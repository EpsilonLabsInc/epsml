import argparse
import json
import torch
import torch.multiprocessing as mp
import yaml

from transformers import DistilBertTokenizer

from epsclassifiers.dino_v3_classifier import DinoV3Classifier
from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.config.config_loader import ConfigLoader
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters
from dino_v3 import DinoV3Type


def main(config_path):
    # Improve tensor IPC robustness for large batches by using file_system sharing.
    try:
        mp.set_sharing_strategy('file_system')
    except Exception:
        pass

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
        max_positive_samples=config["data"]["max_positive_samples"],
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

    # Create the tokenizer (for report text if used).
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") if config["training"]["use_report_text"] else None

    # Create the model.
    print("Creating the model")

    dino_v3_type_map = {
        "small": DinoV3Type.SMALL,
        "base": DinoV3Type.BASE,
        "large": DinoV3Type.LARGE,
        "giant": DinoV3Type.GIANT,
    }
    dino_v3_type_enum = dino_v3_type_map.get(config["training"]["backbone_type"].lower(), DinoV3Type.GIANT)

    model = DinoV3Classifier(
        num_classes=len(dataset_helper.get_labels()),
        dino_v3_checkpoint=config["paths"]["backbone_checkpoint_dir"],
        dino_v3_output_dim=config["training"]["backbone_output_dim"],
        dino_v3_type=dino_v3_type_enum,
        img_size=config["training"]["backbone_img_size"],
        use_attention_pooling=config["training"]["use_attentional_pooling"]
    )

    model = model.to("cuda")
    image_processor = model.get_image_processor()

    for param in model.parameters():
        param.requires_grad = True

    # Freeze backbone.
    for param in model.dino_v3.parameters():
        param.requires_grad = False

    # Prepare the training data.
    print("Preparing the training data")

    # Preprocess images in DataLoader workers by replacing helper datasets.
    from torch.utils.data import Dataset

    class PreprocessedTorchDataset(Dataset):
        def __init__(self, pandas_dataframe, dataset_helper, image_processor, multi_image_input=False, num_multi_images=None):
            self.df = pandas_dataframe
            self.helper = dataset_helper
            self.processor = image_processor
            self.multi_image_input = multi_image_input
            self.num_multi_images = num_multi_images

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            try:
                item = self.df.iloc[idx]
                images = self.helper.get_pil_image(item)
                if not self.multi_image_input:
                    pixel_values = self.processor(images=[images[0]], return_tensors="pt").pixel_values[0].to(torch.float16)
                else:
                    if self.num_multi_images is not None:
                        assert len(images) == self.num_multi_images, f"Expected {self.num_multi_images} images, got {len(images)}"
                        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(torch.float16)
                    else:
                        pixel_values = [
                            self.processor(images=[img], return_tensors="pt").pixel_values[0].to(torch.float16) for img in images
                        ]

                label = self.helper.get_torch_label(item).to(torch.float32)
                data = {
                    "images": pixel_values,
                    "report_texts": None,
                    "text_encodings": None,
                    "file_names": item["relative_image_paths"],
                }
                return data, label
            except:
                return None

    # Try replacing train/val datasets with preprocessed versions
    try:
        train_ds = dataset_helper.get_torch_train_dataset()
        val_ds = dataset_helper.get_torch_validation_dataset()
        train_df = getattr(train_ds, "_GenericTorchDataset__pandas_dataframe", None)
        val_df = getattr(val_ds, "_GenericTorchDataset__pandas_dataframe", None)
        if train_df is not None:
            pre_ds = PreprocessedTorchDataset(
                train_df,
                dataset_helper,
                image_processor,
                config["training"]["multi_image_input"],
                config["training"]["num_multi_images"]
            )
            setattr(dataset_helper, "_GenericDatasetHelper__torch_train_dataset", pre_ds)
            print(f"Replaced train dataset with PreprocessedTorchDataset of length {len(pre_ds)}")
        if val_df is not None:
            pre_val = PreprocessedTorchDataset(
                val_df,
                dataset_helper,
                image_processor,
                config["training"]["multi_image_input"],
                config["training"]["num_multi_images"]
            )
            setattr(dataset_helper, "_GenericDatasetHelper__torch_validation_dataset", pre_val)
            print(f"Replaced val dataset with PreprocessedTorchDataset of length {len(pre_val)}")
    except Exception as e:
        print(f"Could not replace helper datasets with preprocessed variants: {e}")

    training_parameters = TrainingParameters(
        learning_rate=config["training"]["learning_rate"],
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
        save_visualizaton_data_during_training=False,
        save_visualizaton_data_during_validation=False,
        pause_on_validation_visualization=False,
    )

    mlops_parameters = MlopsParameters(
        mlops_type=MlopsType.WANDB,
        experiment_name=config["general"]["experiment_name"],
        run_name=config["general"]["run_name"],
        notes=config["general"]["notes"],
        label_names=dataset_helper.get_labels(),
        send_notification=config["training"]["send_wandb_notification"])

    training_helper = TorchTrainingHelper(
        model=model,
        dataset_helper=dataset_helper,
        device=config["training"]["device"],
        device_ids=config["training"]["device_ids"],
        training_parameters=training_parameters,
        mlops_parameters=mlops_parameters,
        multi_gpu_padding=(config["training"]["multi_image_input"] and config["training"]["num_multi_images"] is None),
        config_file_content=config_str,
    )

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
                pixel_values = pixel_values.to(torch.float32)
                return pixel_values
            except:
                return None

        elif config["training"]["multi_image_input"] and config["training"]["num_multi_images"] is None:
            try:
                image_list = []
                for item in samples:
                    images = dataset_helper.get_pil_image(item)
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(torch.float32)
                    image_list.append(pixel_values)
                return image_list
            except:
                return None

        else:
            try:
                images = [dataset_helper.get_pil_image(item)[0] for item in samples]
                pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(torch.float32)
                return pixel_values
            except:
                return None

    def get_text_encodings(samples):
        report_texts = [dataset_helper.get_report_text(item) for item in samples]
        text_encodings = tokenizer(report_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return report_texts, text_encodings

    def get_torch_labels(samples):
        labels = torch.stack([dataset_helper.get_torch_label(item).to(torch.float32) for item in samples])
        return labels

    def collate_function(samples):
        # Fast path: dataset already preprocessed in workers
        if len(samples) > 0 and isinstance(samples[0], tuple) and isinstance(samples[0][0], dict):
            data_list, labels_list = zip(*samples)
            img0 = data_list[0]["images"]
            # Case A: each sample has a single image tensor
            if isinstance(img0, torch.Tensor):
                images = torch.stack([d["images"] for d in data_list]).to(torch.float32)
            # Case B: each sample has a list of image tensors (multi-image input)
            elif isinstance(img0, list):
                images_list = [d["images"] for d in data_list]
                # If all samples have same number of images and all tensors same shape, stack to [B, N, C, H, W]
                try:
                    n_imgs = len(images_list[0])
                    if all(isinstance(t, torch.Tensor) for t in images_list[0]) and all(len(s)==n_imgs for s in images_list):
                        stacked = [torch.stack([t.to(torch.float32) for t in s]) for s in images_list]
                        images = torch.stack(stacked)  # [B, N, C, H, W]
                    else:
                        # Keep as nested list of tensors
                        images = [[t.to(torch.float32) for t in s if isinstance(t, torch.Tensor)] for s in images_list]
                except Exception:
                    images = [[t.to(torch.float32) for t in s if isinstance(t, torch.Tensor)] for s in images_list]
            else:
                # Unexpected type; fallback to original path below
                images = None

            labels = torch.stack(labels_list).to(torch.float32)
            data = {
                "images": images,
                "report_texts": None,
                "text_encodings": None,
                "file_names": [d["file_names"] for d in data_list],
            }
            return data, labels
        # Images
        images = get_torch_images(samples)
        if images is None:
            return None

        # Text (optional)
        report_texts, text_encodings = get_text_encodings(samples) if config["training"]["use_report_text"] else (None, None)

        # Labels
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
    parser = argparse.ArgumentParser(description="DINOv3 classifier training script")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    main(args.config_path)
