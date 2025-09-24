import yaml


class ConfigLoader:
    def __init__(self):
        pass

    def load_config(self, config_file):
        # Read the configuration file.
        with open(config_file, "r") as file:
            config_file_content = file.read()

        # Load config.
        config = yaml.safe_load(config_file_content)

        # Get the configuration parameters.
        config_dict = {}

        # General section.
        config_dict["general"] = {}
        config_dict["general"]["experiment_name"] = config["general"].get("experiment_name", "")
        config_dict["general"]["run_name"]        = config["general"].get("run_name", "")
        config_dict["general"]["notes"]           = config["general"].get("notes", "")
        config_dict["general"]["save_full_model"] = config["general"].get("save_full_model", False)

        # Paths section.
        config_dict["paths"] = {}
        config_dict["paths"]["backbone_checkpoint_dir"] = config["paths"].get("backbone_checkpoint_dir", "")
        config_dict["paths"]["train_file"]              = config["paths"].get("train_file", "")
        config_dict["paths"]["validation_file"]         = config["paths"].get("validation_file", "")
        config_dict["paths"]["test_file"]               = config["paths"].get("test_file", "")
        config_dict["paths"]["base_path_substitutions"] = self.__convert_none(config["paths"].get("base_path_substitutions", None))
        config_dict["paths"]["output_dir"]              = config["paths"].get("output_dir", "")

        # Data section.
        config_dict["data"] = {}
        config_dict["data"]["body_part"]                      = config["data"].get("body_part", "")
        config_dict["data"]["sub_body_part"]                  = self.__convert_none(config["data"].get("sub_body_part", None))
        config_dict["data"]["custom_labels"]                  = self.__convert_none(config["data"].get("custom_labels", None))
        config_dict["data"]["treat_uncertain_as_positive"]    = config["data"].get("treat_uncertain_as_positive", False)
        config_dict["data"]["perform_label_balancing"]        = config["data"].get("perform_label_balancing", False)
        config_dict["data"]["negative_body_parts_ratio"]      = self.__convert_none(config["data"].get("negative_body_parts_ratio", None))
        config_dict["data"]["num_data_augmentations"]         = config["data"].get("num_data_augmentations", 0)
        config_dict["data"]["compute_num_data_augmentations"] = config["data"].get("compute_num_data_augmentations", False)
        config_dict["data"]["data_augmentation_target"]       = config["data"].get("data_augmentation_target", 0)
        config_dict["data"]["data_augmentation_min"]          = config["data"].get("data_augmentation_min", 0)
        config_dict["data"]["max_study_images"]               = self.__convert_none(config["data"].get("max_study_images", None))
        config_dict["data"]["replace_dicom_with_png"]         = config["data"].get("replace_dicom_with_png", False)

        # Training section.
        config_dict["training"] = {}
        config_dict["training"]["perform_intra_epoch_validation"] = config["training"].get("perform_intra_epoch_validation", False)
        config_dict["training"]["intra_epoch_validation_step"]    = config["training"].get("intra_epoch_validation_step", 5000)
        config_dict["training"]["send_wandb_notification"]    = config["training"].get("send_wandb_notification", False)
        config_dict["training"]["device"]                         = config["training"].get("device", "")
        config_dict["training"]["device_ids"]                     = self.__convert_none(config["training"].get("device_ids", None))
        config_dict["training"]["num_training_workers_per_gpu"]   = config["training"].get("num_training_workers_per_gpu", 1)
        config_dict["training"]["num_validation_workers_per_gpu"] = config["training"].get("num_validation_workers_per_gpu", 1)
        config_dict["training"]["learning_rate"]                  = config["training"].get("learning_rate", 1e-6)
        config_dict["training"]["warmup_ratio"]                   = config["training"].get("warmup_ratio", 0.1)
        config_dict["training"]["num_epochs"]                     = config["training"].get("num_epochs", 1)
        config_dict["training"]["training_batch_size"]            = config["training"].get("training_batch_size", 1)
        config_dict["training"]["validation_batch_size"]          = config["training"].get("validation_batch_size", 1)
        config_dict["training"]["min_allowed_batch_size"]         = config["training"].get("min_allowed_batch_size", 1)
        config_dict["training"]["use_report_text"]                = config["training"].get("use_report_text", False)
        config_dict["training"]["multi_image_input"]              = config["training"].get("multi_image_input", False)
        config_dict["training"]["num_multi_images"]               = self.__convert_none(config["training"].get("num_multi_images", None))
        config_dict["training"]["use_attentional_pooling"]        = config["training"].get("use_attentional_pooling", False)

        return config_dict

    def __convert_none(self, value):
        if value == "None":
            return None

        return value
