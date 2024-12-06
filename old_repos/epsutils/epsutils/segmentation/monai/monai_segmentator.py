import os

import numpy as np
import torch
from monai.bundle import ConfigParser


class MonaiSegmentator:
    def __init__(self):
        self.__base_dir = os.path.dirname(os.path.abspath(__file__))
        self.__config_path = os.path.join(self.__base_dir, "configs/inference.json")
        self.__model_path = os.path.join(self.__base_dir, "models/model.pt")

        # Get config.
        print("Reading Monai configuration")
        self.__config = ConfigParser()
        self.__config.read_config(self.__config_path)
        self.__preprocessing = self.__config.get_parsed_content("preprocessing")
        self.__inferer = self.__config.get_parsed_content("inferer")
        self.__postprocessing = self.__config.get_parsed_content("postprocessing")

        # Load the model.
        print("Loading Monai model")
        self.__model = self.__config.get_parsed_content("network")
        self.__model.load_state_dict(torch.load(self.__model_path))
        self.__model.cuda()
        self.__model.eval()

    def run_segmentation_pipeline(self, file_or_dir):
        data = self.preprocessing(file_or_dir=file_or_dir)
        result = self.segmentation(data=data)
        return result

    def preprocessing(self, file_or_dir):
        data = self.__preprocessing({"image": file_or_dir})
        return data

    def segmentation(self, data):
        # Add batch dimension and move to GPU.
        data["image"] = data["image"].unsqueeze(0).cuda()

        # Run inference.
        with torch.no_grad():
            data["pred"] = self.__inferer(data["image"], network=self.__model)

        # Remove batch dimension.
        data["image"] = data["image"][0]
        data["pred"] = data["pred"][0]

        # Move image back to CPU to have more GPU mem for postprocessing.
        image_cpu = data["image"].cpu()
        del data["image"]
        torch.cuda.empty_cache()
        data["image"] = image_cpu

        # Postprocessing.
        data = self.__postprocessing(data)

        # Move segmentation results to CPU.
        segmentation = data["pred"][0].cpu().numpy()

        # Get segmentation info.
        info = self.__get_segmentation_info(segmentation=segmentation)

        return {"segmentation": segmentation, "info": info}

    def __get_segmentation_info(self, segmentation):
        info = {}

        labels, counts = np.unique(segmentation, return_counts=True)
        labels = labels.astype(int)
        non_zero_labels = labels != 0  # Exclude background.
        labels = labels[non_zero_labels]
        counts = counts[non_zero_labels]
        top_label = labels[np.argmax(counts)]

        info["all_labels"] = labels.tolist()
        info["label_counts"] = counts.tolist()
        info["top_label"] = top_label

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
