import copy
from typing import List, Dict

import torch

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils

from intern_vit_classifier import InternVitClassifier


class InternVitCompositeBinaryClassifier:
    def __init__(self,
                 binary_classifier_checkpoints: List[Dict[str, str]],
                 intern_vl_checkpoint_dir,
                 intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                 multi_image_input=True,
                 num_multi_images=2,
                 device="cuda",
                 **kwargs):

        self.__multi_image_input = multi_image_input
        self.__num_multi_images = num_multi_images
        self.__device = device

        # At least one binary classifier is required.
        assert len(binary_classifier_checkpoints) > 0

        # Create InternViT classifier.
        print("Creating InternViT classifier")
        self.__intern_vit_classifier = InternVitClassifier(num_classes=1,
                                                           intern_vl_checkpoint_dir=intern_vl_checkpoint_dir,
                                                           intern_vit_output_dim=intern_vit_output_dim,
                                                           multi_image_input=multi_image_input,
                                                           num_multi_images=num_multi_images,
                                                           **kwargs)

        self.__intern_vit_classifier.to(self.__device)
        self.__intern_vit_classifier.eval()

        # Load binary classifier checkpoints.
        self.__binary_classifiers = []
        for checkpoint in binary_classifier_checkpoints:
            checkpoint_name = checkpoint["name"]
            checkpoint_path = checkpoint["path"]
            print(f"Loading binary classifier checkpoint for '{checkpoint_name}'")

            state_dict = torch.load(checkpoint_path)
            self.__intern_vit_classifier.load_state_dict(state_dict["model_state_dict"])
            binary_classifier = copy.deepcopy(self.__intern_vit_classifier.classifier)
            binary_classifier.eval()
            self.__binary_classifiers.append({"name": checkpoint_name, "model": binary_classifier})

    def predict(self, dicom_files):
        if self.__multi_image_input:
            assert len(dicom_files) == self.__num_multi_images
        else:
            assert len(dicom_files) == 1

        # Convert DICOM files to PIL images.
        images = []
        for dicom_file in dicom_files:
            image = dicom_utils.get_dicom_image(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
            images.append(image)

        # Preprocess images and convert to bfloat16.
        pixel_values = self.__intern_vit_classifier.get_image_processor()(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(torch.bfloat16)

        # Make a batch of size 2 since InternVitClassifier currently supports only batch sizes >= 2.
        pixel_values = torch.stack([pixel_values, pixel_values])

        # Run prediction using InternViT classifier.
        with torch.no_grad():
            res = self.__intern_vit_classifier(pixel_values.to(self.__device))
            embeddings = res["embeddings"]

        # Pass embeddings into separate binary classifiers.
        with torch.no_grad():
            all_probs = []
            for classifier in self.__binary_classifiers:
                output = classifier["model"](embeddings)[0]
                probs = torch.sigmoid(output)
                all_probs.append({"name": classifier["name"], "probs": probs})

        # Sanity check.
        assert torch.allclose(all_probs[-1]["probs"], torch.sigmoid(res["output"]), atol=1e-6)

        return all_probs

    def get_image_processor(self):
        return self.__intern_vit_classifier.get_image_processor()


if __name__ == "__main__":
    # InternVitCompositeBinaryClassifier example.

    INTERN_VL_CHECKPOINT_DIR = "/mnt/efs/models/internvl/old/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
    BINARY_CLASSIFIER_CHECKPOINTS = [
        {"name": "consolidation", "path": "/home/andrej/tmp/binary_checkpoints/binary_consolidation_checkpoint.pt"},
        {"name": "edema", "path": "/home/andrej/tmp/binary_checkpoints/binary_edema_checkpoint.pt"},
        {"name": "airspace opacity", "path": "/home/andrej/tmp/binary_checkpoints/binary_airspace_opacity_checkpoint.pt"},
    ]

    # Edema.
    DICOM_FILES = [
        "/mnt/efs/all-cxr/gradient/22JUL2024/GRDN87VZUPS8K1XS/GRDNJ7GNGOQI87RV/studies/1.2.826.0.1.3680043.8.498.43077057944350341201820145574876698829/series/1.2.826.0.1.3680043.8.498.71619914322538516903099898284356295253/instances/1.2.826.0.1.3680043.8.498.13418427961977328036029752570867031541.dcm",
        "/mnt/efs/all-cxr/gradient/22JUL2024/GRDN87VZUPS8K1XS/GRDNJ7GNGOQI87RV/studies/1.2.826.0.1.3680043.8.498.43077057944350341201820145574876698829/series/1.2.826.0.1.3680043.8.498.96671321347557557720154152227531287229/instances/1.2.826.0.1.3680043.8.498.83519664942235257620802927312300077032.dcm"
    ]

    # Consolidation.
    # DICOM_FILES = [
    #     "/mnt/efs/all-cxr/gradient/20DEC2024/deid/GRDN00SZ4W5JTK30/GRDNV9MCKLV7SYIM/studies/1.2.826.0.1.3680043.8.498.24776930674676137805272112279298177654/series/1.2.826.0.1.3680043.8.498.68343223226858939668117029668988077247/instances/1.2.826.0.1.3680043.8.498.38734824056281407652077901128530906919.dcm",
    #     "/mnt/efs/all-cxr/gradient/20DEC2024/deid/GRDN00SZ4W5JTK30/GRDNV9MCKLV7SYIM/studies/1.2.826.0.1.3680043.8.498.24776930674676137805272112279298177654/series/1.2.826.0.1.3680043.8.498.33972695614151366813675462694932878085/instances/1.2.826.0.1.3680043.8.498.13892220803373546304991929643955934631.dcm"
    # ]

    # Airspace opacity.
    # DICOM_FILES = [
    #     "/mnt/efs/all-cxr/gradient/22JUL2024/GRDN56VW2AEIGSD3/GRDN8OLYZSVQH71V/studies/1.2.826.0.1.3680043.8.498.52927558507360759327732750959950731826/series/1.2.826.0.1.3680043.8.498.42596855183528922700105748925390838125/instances/1.2.826.0.1.3680043.8.498.30130378567856671404562122987382276294.dcm",
    #     "/mnt/efs/all-cxr/gradient/22JUL2024/GRDN56VW2AEIGSD3/GRDN8OLYZSVQH71V/studies/1.2.826.0.1.3680043.8.498.52927558507360759327732750959950731826/series/1.2.826.0.1.3680043.8.498.55470003897800545694532330351011469070/instances/1.2.826.0.1.3680043.8.498.41286554462632571993772012924613327790.dcm"
    # ]

    print("Running InternVitCompositeBinaryClassifier example")

    classifier = InternVitCompositeBinaryClassifier(
        binary_classifier_checkpoints=BINARY_CLASSIFIER_CHECKPOINTS,
        intern_vl_checkpoint_dir=INTERN_VL_CHECKPOINT_DIR,
        intern_vit_output_dim=3200,
        multi_image_input=True,
        num_multi_images=2)

    output = classifier.predict(dicom_files=DICOM_FILES)

    print("Predicted values:")
    print(output)
