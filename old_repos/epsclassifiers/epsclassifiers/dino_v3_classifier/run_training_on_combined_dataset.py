import argparse
import json
import torch
import torch.multiprocessing as mp
import yaml

from transformers import DistilBertTokenizer

from epsclassifiers.dino_v3_classifier import DinoV3Classifier
from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters
from dino_v3 import DinoV3Type


def _convert_none(value):
    if value == "None":
        return None
    return value


def main(config_path):
    # Improve tensor IPC robustness for large batches by using file_system sharing.
    try:
        mp.set_sharing_strategy('file_system')
    except Exception:
        pass

    # Load config
    with open(config_path, "r") as file:
        config_file_content = file.read()
    config = yaml.safe_load(config_file_content)

    # Print configuration parameters succinctly (like Intern)
    print("----------------------------------------------------------")
    print("Using the following configuration parameters:")
    print(json.dumps(config, indent=4))
    print("----------------------------------------------------------")

    # Derived names (keep current format)
    experiment_name = f"{config['general']['model_name']}-training-on-{config['general']['dataset_name']}"
    experiment_dir = f"{config['paths']['output_dir']}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"
    
    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = GenericDatasetHelper(
        train_file=config["paths"]["train_file"],
        validation_file=config["paths"]["validation_file"],
        test_file=config["paths"]["test_file"],
        base_path_substitutions=config["paths"].get("base_path_substitutions"),
        body_part=config["general"]["body_part"],
        sub_body_part=config["general"].get("sub_body_part"),
        merge_val_and_test=True,
        treat_uncertain_as_positive=config["general"].get("treat_uncertain_as_positive", False),
        perform_label_balancing=config["general"].get("perform_label_balancing", False),
        num_data_augmentations=config["general"].get("num_data_augmentations", 0),
        compute_num_data_augmentations=config["general"].get("compute_num_data_augmentations", False),
        data_augmentation_target=config["general"].get("data_augmentation_target", 0),
        data_augmentation_min=config["general"].get("data_augmentation_min", 0),
        unroll_images=config["general"].get("unroll_images", True),
        max_study_images=(
            config["general"].get("max_study_images_to_unroll")
            if "max_study_images_to_unroll" in config.get("general", {})
            else config["training"].get("max_multi_images")
        ),
        convert_images_to_rgb=True,
        replace_dicom_with_png=config["general"].get("replace_dicom_with_png", False),
        custom_labels=config["general"].get("custom_labels"))
    
    print(f"Using the following labels: {dataset_helper.get_labels()}")
    
    # Create the tokenizer (for report text if used).
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") if config["general"].get("use_report_text", False) else None
    
    # Create the model.
    print("Creating the model")
    
    dino_v3_type_map = {
        "small": DinoV3Type.SMALL,
        "base": DinoV3Type.BASE,
        "large": DinoV3Type.LARGE,
        "giant": DinoV3Type.GIANT,
    }
    dino_v3_type_enum = dino_v3_type_map.get(config["general"].get("dino_v3_type", "giant").lower(), DinoV3Type.GIANT)

    model = DinoV3Classifier(
        num_classes=len(dataset_helper.get_labels()),
        dino_v3_checkpoint=config["paths"].get("dino_v3_checkpoint_path") or None,
        dino_v3_output_dim=config["general"].get("dino_v3_output_dim", 4096),
        dino_v3_type=dino_v3_type_enum,
        img_size=config["general"].get("dino_v3_img_size", 1024),
        use_attention_pooling=config["general"].get("use_attention_pooling", False),
    )
    
    model = model.to("cuda")
    image_processor = model.get_image_processor()
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Freeze backbone.
    for param in model.dino_v3.parameters():
        param.requires_grad = False

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
                config["training"].get("multi_image_input", False),
                _convert_none(config["training"].get("num_multi_images")),
            )
            setattr(dataset_helper, "_GenericDatasetHelper__torch_train_dataset", pre_ds)
            print(f"Replaced train dataset with PreprocessedTorchDataset of length {len(pre_ds)}")
        if val_df is not None:
            pre_val = PreprocessedTorchDataset(
                val_df,
                dataset_helper,
                image_processor,
                config["training"].get("multi_image_input", False),
                _convert_none(config["training"].get("num_multi_images")),
            )
            setattr(dataset_helper, "_GenericDatasetHelper__torch_validation_dataset", pre_val)
            print(f"Replaced val dataset with PreprocessedTorchDataset of length {len(pre_val)}")
    except Exception as e:
        print(f"Could not replace helper datasets with preprocessed variants: {e}")
    
    training_parameters = TrainingParameters(
        learning_rate=config["training"].get("learning_rate", 1e-6),
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        num_epochs=config["training"].get("num_epochs", 1),
        training_batch_size=config["training"].get("training_batch_size", 1),
        validation_batch_size=config["training"].get("validation_batch_size", 1),
        min_allowed_batch_size=config["training"].get("min_allowed_batch_size", 2),
        criterion=SampleBalancedBCEWithLogitsLoss(),
        checkpoint_dir=checkpoint_dir,
        perform_intra_epoch_validation=config["training"].get("perform_intra_epoch_validation", False),
        intra_epoch_validation_step=config["training"].get("intra_epoch_validation_step", 5000),
        num_training_workers_per_gpu=config["training"].get("num_training_workers_per_gpu", 1),
        num_validation_workers_per_gpu=config["training"].get("num_validation_workers_per_gpu", 1),
        save_visualizaton_data_during_training=True,
        save_visualizaton_data_during_validation=True,
        pause_on_validation_visualization=False,
    )
    
    mlops_parameters = MlopsParameters(
        mlops_type=MlopsType.WANDB,
        experiment_name=experiment_name,
        run_name=config["general"].get("run_name", ""),
        notes=config["general"].get("notes", ""),
        label_names=dataset_helper.get_labels(),
        send_notification=config["training"].get("send_wandb_notification", True),
    )
    
    training_helper = TorchTrainingHelper(
        model=model,
        dataset_helper=dataset_helper,
        device=config["training"].get("device", ""),
        device_ids=_convert_none(config["training"].get("device_ids")),
        training_parameters=training_parameters,
        mlops_parameters=mlops_parameters,
        multi_gpu_padding=(config["training"].get("multi_image_input", False) and config["training"].get("num_multi_images") is None),
        config_file_content=config_file_content,
    )
    
    device_ids_used = training_helper.get_device_ids_used()
    
    def get_torch_images(samples):
        if config["training"].get("multi_image_input", False) and _convert_none(config["training"].get("num_multi_images")) is not None:
            try:
                stack = []
                for item in samples:
                    images = dataset_helper.get_pil_image(item)
                    assert len(images) == _convert_none(config["training"].get("num_multi_images"))
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    stack.append(pixel_values)
                
                pixel_values = torch.stack(stack)
                pixel_values = pixel_values.to(torch.float32)
                return pixel_values
            except:
                return None
        
        elif config["training"].get("multi_image_input", False) and _convert_none(config["training"].get("num_multi_images")) is None:
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
        report_texts, text_encodings = get_text_encodings(samples) if config["general"].get("use_report_text", False) else (None, None)

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
    
    if config["general"].get("save_full_model", False):
        training_helper.save_model(model_file_name=save_model_filename, parallel_model_file_name=save_parallel_model_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv3 classifier training script")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args.config_path)
