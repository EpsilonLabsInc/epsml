import copy
from typing import List, Dict

import torch

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.training.probabilities_reduction import ProbabilitiesReductionStrategy, probabilities_reduction

from intern_vit_classifier import (
    AttentionalPoolingWithClassifierHead,
    InternVitClassifier,
)


class InternVitCompositeBinaryClassifier:
    def __init__(self,
                 grouped_binary_classifier_checkpoints,
                 intern_vl_checkpoint_dir,
                 intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                 device="cuda",
                 probabilities_reduction_strategy=ProbabilitiesReductionStrategy.MAX,
                 use_attentional_pooling=True,
                 **kwargs):

        self.__grouped_binary_classifier_checkpoints = grouped_binary_classifier_checkpoints
        self.__intern_vl_checkpoint_dir = intern_vl_checkpoint_dir
        self.__intern_vit_output_dim = intern_vit_output_dim
        self.__device = device
        self.__probabilities_reduction_strategy = probabilities_reduction_strategy
        self.__use_attentional_pooling = use_attentional_pooling

        self.__backbones = {}
        self.__heads = {}

        for group in grouped_binary_classifier_checkpoints:
            num_multi_images = grouped_binary_classifier_checkpoints[group]["num_multi_images"]
            checkpoints = grouped_binary_classifier_checkpoints[group]["checkpoints"]
            backbone = self.__get_backbone(num_multi_images)

            self.__heads[group] = {
                "backbone": backbone,
                "num_multi_images": num_multi_images,
                "heads": []
            }

            for checkpoint in checkpoints:
                name = checkpoint["name"]
                path = checkpoint["path"]

                print(f"Loading binary classifier checkpoint for '{group}/{name}'")
                state_dict = torch.load(path)
                backbone.load_state_dict(state_dict["model_state_dict"])
                if self.__use_attentional_pooling:
                    head = AttentionalPoolingWithClassifierHead(
                        copy.deepcopy(backbone.attentional_pooling), copy.deepcopy(backbone.classifier)
                    )
                else:
                    head = copy.deepcopy(backbone.classifier)
                head.to(self.__device)
                head.eval()
                self.__heads[group]["heads"].append({"name": name, "model": head})

    def predict(self, group, dicom_files):
        num_multi_images = self.__heads[group]["num_multi_images"]

        # In multi-image mode, the number of input images must match (or be None to indicate attention pooling).
        assert num_multi_images == 1 or len(dicom_files) == num_multi_images or num_multi_images is None

        # Convert DICOM files to PIL images.
        images = []
        for dicom_file in dicom_files:
            # TODO: Handle compressed DICOM files.
            image = dicom_utils.get_dicom_image_fail_safe(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
            images.append(image)

        # Preprocess images and convert to bfloat16.
        backbone = self.__heads[group]["backbone"]
        pixel_values = backbone.get_image_processor()(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(torch.bfloat16)

        # Add batch dimension.
        if num_multi_images is None or num_multi_images > 1:
            pixel_values = pixel_values.unsqueeze(0)

        # Run prediction using the backbone.
        with torch.no_grad():
            res = backbone(pixel_values.to(self.__device))
            embeddings = res["embeddings"]
            last_hidden_state = res["last_hidden_state"]

        # Pass embeddings into separate heads.
        with torch.no_grad():
            all_probs = []
            for head in self.__heads[group]["heads"]:
                if self.__use_attentional_pooling:
                    output = head["model"](last_hidden_state)
                else:
                    output = head["model"](embeddings)
                probs = torch.sigmoid(output)

                if num_multi_images == 1:
                    probs = probabilities_reduction(probs, self.__probabilities_reduction_strategy)

                all_probs.append({"name": head["name"], "probs": probs})

        return all_probs

    def __get_backbone(self, num_multi_images: int | None):
        if num_multi_images in self.__backbones:
            return self.__backbones[num_multi_images]

        print(f"Creating InternViT classifier backbone with {num_multi_images} image input")
        backbone = InternVitClassifier(num_classes=1,
                                       intern_vl_checkpoint_dir=self.__intern_vl_checkpoint_dir,
                                       intern_vit_output_dim=self.__intern_vit_output_dim,
                                       multi_image_input=(num_multi_images > 1) if num_multi_images is not None else True,
                                       num_multi_images=num_multi_images,
                                       use_attentional_pooling=self.__use_attentional_pooling)

        backbone.to(self.__device)
        backbone.eval()
        self.__backbones[num_multi_images] = backbone

        return backbone


if __name__ == "__main__":
    # InternVitCompositeBinaryClassifier example.

    INTERN_VL_CHECKPOINT_DIR = "/mnt/training/internvl_weights/internvl3_chimera_20250810_083849_1e-5_0810_no_label_gpt_bodypart/checkpoint-10466"
    GROUPED_BINARY_CLASSIFIER_CHECKPOINTS = {
        "chest": {
            "num_multi_images": 2,
            "checkpoints": [
                {"name": "emphysema", "path": "/mnt/training/attention_pooling_checkpoints/chest/release_intern_vit_classifier-training-on-chest_emphysema/checkpoint/checkpoint_epoch_1_20250828_122548_utc.pt"},
                {"name": "support_devices", "path": "/release_intern_vit_classifier-training-on-chest_support_devices/checkpoint/checkpoint_epoch_1_20250827_155454_utc.pt"},
            ]
        },
        "head": {
            "num_multi_images": None,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/attention_pooling_checkpoints/head/release_intern_vit_classifier-training-on-chest_emphysema/checkpoint/checkpoint_epoch_1_20250828_122548_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/attention_pooling_checkpoints/head/release_intern_vit_classifier-training-on-chest_emphysema/checkpoint/checkpoint_epoch_1_20250828_122548_utc.pt"},
            ]
        },
        "pelvis": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/attention_pooling_checkpoints/pelvis/release_intern_vit_classifier-training-on-chest_emphysema/checkpoint/checkpoint_epoch_1_20250828_122548_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/attention_pooling_checkpoints/pelvis/release_intern_vit_classifier-training-on-chest_emphysema/checkpoint/checkpoint_epoch_1_20250828_122548_utc.pt"},
            ]
        },
    }

    # Edema.
    DICOM_FILES = [
        "/mnt/sfs-gradient-chest/22JUL2024/GRDN87VZUPS8K1XS/GRDNJ7GNGOQI87RV/studies/1.2.826.0.1.3680043.8.498.43077057944350341201820145574876698829/series/1.2.826.0.1.3680043.8.498.71619914322538516903099898284356295253/instances/1.2.826.0.1.3680043.8.498.13418427961977328036029752570867031541.dcm",
        "/mnt/sfs-gradient-chest/22JUL2024/GRDN87VZUPS8K1XS/GRDNJ7GNGOQI87RV/studies/1.2.826.0.1.3680043.8.498.43077057944350341201820145574876698829/series/1.2.826.0.1.3680043.8.498.96671321347557557720154152227531287229/instances/1.2.826.0.1.3680043.8.498.83519664942235257620802927312300077032.dcm"
    ]

    # Consolidation.
    # DICOM_FILES = [
    #     "/mnt/sfs-gradient-chest/20DEC2024/GRDN00SZ4W5JTK30/GRDNV9MCKLV7SYIM/studies/1.2.826.0.1.3680043.8.498.24776930674676137805272112279298177654/series/1.2.826.0.1.3680043.8.498.68343223226858939668117029668988077247/instances/1.2.826.0.1.3680043.8.498.38734824056281407652077901128530906919.dcm",
    #     "/mnt/sfs-gradient-chest/20DEC2024/GRDN00SZ4W5JTK30/GRDNV9MCKLV7SYIM/studies/1.2.826.0.1.3680043.8.498.24776930674676137805272112279298177654/series/1.2.826.0.1.3680043.8.498.33972695614151366813675462694932878085/instances/1.2.826.0.1.3680043.8.498.13892220803373546304991929643955934631.dcm"
    # ]

    # Airspace opacity.
    # DICOM_FILES = [
    #     "/mnt/sfs-gradient-chest/22JUL2024/GRDN56VW2AEIGSD3/GRDN8OLYZSVQH71V/studies/1.2.826.0.1.3680043.8.498.52927558507360759327732750959950731826/series/1.2.826.0.1.3680043.8.498.42596855183528922700105748925390838125/instances/1.2.826.0.1.3680043.8.498.30130378567856671404562122987382276294.dcm",
    #     "/mnt/sfs-gradient-chest/22JUL2024/GRDN56VW2AEIGSD3/GRDN8OLYZSVQH71V/studies/1.2.826.0.1.3680043.8.498.52927558507360759327732750959950731826/series/1.2.826.0.1.3680043.8.498.55470003897800545694532330351011469070/instances/1.2.826.0.1.3680043.8.498.41286554462632571993772012924613327790.dcm"
    # ]

    print("Running InternVitCompositeBinaryClassifier example")

    classifier = InternVitCompositeBinaryClassifier(
        grouped_binary_classifier_checkpoints=GROUPED_BINARY_CLASSIFIER_CHECKPOINTS,
        intern_vl_checkpoint_dir=INTERN_VL_CHECKPOINT_DIR,
        intern_vit_output_dim=3200,
        device="cuda",
        use_attentional_pooling=True)

    output = {}
    output["chest"] = classifier.predict(group="chest", dicom_files=DICOM_FILES)
    output["head"] = classifier.predict(group="head", dicom_files=DICOM_FILES)
    output["pelvis"] = classifier.predict(group="pelvis", dicom_files=DICOM_FILES)

    import pprint
    print("Predicted values:")
    pprint.pprint(output, indent=4)
