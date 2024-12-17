import json
import os

import numpy as np
import torch
from monai.bundle import ConfigParser


class MonaiSegmentator:
    def __init__(self, include_labels_distribution=False):
        # Get config.
        print("Reading Monai configuration")
        self.__config = ConfigParser()
        self.__config.read_config(self.get_config_path())
        self.__preprocessing = self.__config.get_parsed_content("preprocessing")
        self.__inferer = self.__config.get_parsed_content("inferer")
        self.__postprocessing = self.__config.get_parsed_content("postprocessing")

        # Load the model.
        print("Loading Monai model")
        self.__model = self.__config.get_parsed_content("network")
        self.__model.load_state_dict(torch.load(self.get_model_path()))
        self.__model.cuda()
        self.__model.eval()

        self.__include_labels_distribution = include_labels_distribution

    @staticmethod
    def get_config_path():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "configs/inference.json")
        return config_path

    @staticmethod
    def get_meta_data_path():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        meta_data_path = os.path.join(base_dir, "configs/metadata.json")
        return meta_data_path

    @staticmethod
    def get_model_path():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models/model.pt")
        return model_path

    @staticmethod
    def get_labels():
        with open(MonaiSegmentator.get_meta_data_path(), "r") as file:
            data = json.load(file)

        labels = data["network_data_format"]["outputs"]["pred"]["channel_def"]
        return labels

    def run_segmentation_pipeline(self, file_or_dir, run_postprocessing_on_gpu=True):
        data = self.preprocessing(file_or_dir=file_or_dir)
        result = self.segmentation(data=data, run_postprocessing_on_gpu=run_postprocessing_on_gpu)
        return result

    def preprocessing(self, file_or_dir):
        data = self.__preprocessing({"image": file_or_dir})
        return data

    def segmentation(self, data, run_postprocessing_on_gpu=True):
        # Add batch dimension and move to GPU.
        data["image"] = data["image"].unsqueeze(0).cuda()

        # Run inference.
        with torch.no_grad():
            data["pred"] = self.__inferer(data["image"], network=self.__model)

        # Remove batch dimension.
        data["image"] = data["image"][0]
        data["pred"] = data["pred"][0]

        # Move prediction to CPU?
        if not run_postprocessing_on_gpu:
            data["pred"] = data["pred"].cpu()

        # Postprocessing.
        data = self.__postprocessing(data)

        # Move segmentation results to CPU if they're not already on CPU.
        segmentation = data["pred"][0].cpu().numpy()

        # Get segmentation info.
        info = self.__get_segmentation_info(segmentation=segmentation)

        # Expicitly free GPU memory.
        del data
        torch.cuda.empty_cache()

        return {"segmentation": segmentation, "info": info}

    def __get_segmentation_info(self, segmentation):
        info = {
            "all_labels": [],
            "label_counts": [],
            "top_label": None,
            "labels_distribution": None
        }

        labels, counts = np.unique(segmentation, return_counts=True)
        labels = labels.astype(int)
        non_zero_labels = labels != 0  # Exclude background.
        labels = labels[non_zero_labels]
        counts = counts[non_zero_labels]

        if labels.size == 0:
            return info

        top_label = labels[np.argmax(counts)]
        info["all_labels"] = labels.tolist()
        info["label_counts"] = counts.tolist()
        info["top_label"] = top_label

        if self.__include_labels_distribution:
            labels_distribution = {}
            num_slices = segmentation.shape[2]
            for slice_index in range(num_slices):
                labels, counts = np.unique(segmentation[:, :, slice_index], return_counts=True)
                labels = labels.astype(int)
                non_zero_labels = labels != 0  # Exclude background.
                labels = labels[non_zero_labels]
                counts = counts[non_zero_labels]
                labels_distribution[slice_index] = {label: count for label, count in zip(labels, counts)}

            info["labels_distribution"] = labels_distribution

        return info
