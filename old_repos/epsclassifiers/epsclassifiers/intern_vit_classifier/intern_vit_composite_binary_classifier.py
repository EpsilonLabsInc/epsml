import copy
from typing import List, Dict

import torch

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.training.probabilities_reduction import ProbabilitiesReductionStrategy, probabilities_reduction

from intern_vit_classifier import InternVitClassifier


class InternVitCompositeBinaryClassifier:
    def __init__(self,
                 grouped_binary_classifier_checkpoints,
                 intern_vl_checkpoint_dir,
                 intern_vit_output_dim=3200,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                 device="cuda",
                 probabilities_reduction_strategy=ProbabilitiesReductionStrategy.MAX,
                 **kwargs):

        self.__grouped_binary_classifier_checkpoints = grouped_binary_classifier_checkpoints
        self.__intern_vl_checkpoint_dir = intern_vl_checkpoint_dir
        self.__intern_vit_output_dim = intern_vit_output_dim
        self.__device = device
        self.__probabilities_reduction_strategy = probabilities_reduction_strategy

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
                head = copy.deepcopy(backbone.classifier)
                head.eval()
                self.__heads[group]["heads"].append({"name": name, "model": head})

    def predict(self, group, dicom_files):
        num_multi_images = self.__heads[group]["num_multi_images"]

        # In multi-image mode, the number of input images must match.
        assert num_multi_images == 1 or len(dicom_files) == num_multi_images

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
        if num_multi_images > 1:
            pixel_values = pixel_values.unsqueeze(0)

        # Run prediction using the backbone.
        with torch.no_grad():
            res = backbone(pixel_values.to(self.__device))
            embeddings = res["embeddings"]

        # Pass embeddings into separate heads.
        with torch.no_grad():
            all_probs = []
            for head in self.__heads[group]["heads"]:
                output = head["model"](embeddings)
                probs = torch.sigmoid(output)

                if num_multi_images == 1:
                    probs = probabilities_reduction(probs, self.__probabilities_reduction_strategy)

                all_probs.append({"name": head["name"], "probs": probs})

        return all_probs

    def __get_backbone(self, num_multi_images):
        if num_multi_images in self.__backbones:
            return self.__backbones[num_multi_images]

        print(f"Creating InternViT classifier backbone with {num_multi_images} image input")
        backbone = InternVitClassifier(num_classes=1,
                                       intern_vl_checkpoint_dir=self.__intern_vl_checkpoint_dir,
                                       intern_vit_output_dim=self.__intern_vit_output_dim,
                                       multi_image_input=num_multi_images > 1,
                                       num_multi_images=num_multi_images)

        backbone.to(self.__device)
        backbone.eval()
        self.__backbones[num_multi_images] = backbone

        return backbone


if __name__ == "__main__":
    # InternVitCompositeBinaryClassifier example.

    INTERN_VL_CHECKPOINT_DIR = "/mnt/training/internvl3_chimera_20250609_233409_1e-5_epsilon_all_0608/checkpoint-24934/"
    GROUPED_BINARY_CLASSIFIER_CHECKPOINTS = {
        "chest": {
            "num_multi_images": 2,
            "checkpoints": [
                {"name": "bronchitis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_bronchitis/checkpoint/checkpoint_epoch_2_20250622_000855_utc.pt"},
                {"name": "adenopathy", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_adenopathy/checkpoint/checkpoint_epoch_2_20250624_193752_utc.pt"},
                {"name": "hyperinflation", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_hyperinflation/checkpoint/checkpoint_epoch_2_20250623_151942_utc.pt"},
                {"name": "scarring", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_scarring/checkpoint/checkpoint_epoch_2_20250623_105825_utc.pt"},
                {"name": "calcification", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_calcification/checkpoint/checkpoint_epoch_2_20250626_181726_utc.pt"},
                {"name": "congestive_heart_failure", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_congestive_heart_failure/checkpoint/checkpoint_epoch_2_20250624_183840_utc.pt"},
                {"name": "cardiomegaly", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_cardiomegaly/checkpoint/checkpoint_epoch_2_20250624_075643_utc.pt"},
                {"name": "fibrosis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_fibrosis/checkpoint/checkpoint_epoch_2_20250622_085746_utc.pt"},
                {"name": "enlarged_mediastinum", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_enlarged_mediastinum/checkpoint/checkpoint_epoch_2_20250624_190344_utc.pt"},
                {"name": "hernia", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_hernia/checkpoint/checkpoint_epoch_2_20250624_203328_utc.pt"},
                {"name": "foreign_bodies", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_foreign_bodies/checkpoint/checkpoint_epoch_2_20250625_185744_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_support_devices/checkpoint/checkpoint_epoch_2_20250625_211449_utc.pt"},
                {"name": "pneumonitis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pneumonitis/checkpoint/checkpoint_epoch_2_20250630_184602_utc.pt"},
                {"name": "tortuous_aorta", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_tortuous_aorta/checkpoint/checkpoint_epoch_2_20250622_145510_utc.pt"},
                {"name": "pleural_effusion", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pleural_effusion/checkpoint/checkpoint_epoch_2_20250623_160141_utc.pt"},
                {"name": "sarcoidosis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_sarcoidosis/checkpoint/checkpoint_epoch_2_20250625_190616_utc.pt"},
                {"name": "granuloma", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_granuloma/checkpoint/checkpoint_epoch_2_20250620_214813_utc.pt"},
                {"name": "consolidation", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_consolidation/checkpoint/checkpoint_epoch_2_20250620_202144_utc.pt"},
                {"name": "pulmonary_cavity", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pulmonary_cavity/checkpoint/checkpoint_epoch_2_20250625_193529_utc.pt"},
                {"name": "enlarged_pulmonary_artery", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_enlarged_pulmonary_artery/checkpoint/checkpoint_epoch_2_20250621_032501_utc.pt"},
                {"name": "pericardial_effusion", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pericardial_effusion/checkpoint/checkpoint_epoch_2_20250630_163716_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_no_findings/checkpoint/checkpoint_epoch_2_20250627_191151_utc.pt"},
                {"name": "asthma", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_asthma/checkpoint/checkpoint_epoch_2_20250707_181857_utc.pt"},
                {"name": "covid_19", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_covid_19/checkpoint/checkpoint_epoch_2_20250703_175539_utc.pt"},
                {"name": "arthritis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_arthritis/checkpoint/checkpoint_epoch_2_20250621_225724_utc.pt"},
                {"name": "pneumothorax", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pneumothorax/checkpoint/checkpoint_epoch_2_20250622_093109_utc.pt"},
                {"name": "mass_nodule", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_mass_nodule/checkpoint/checkpoint_epoch_2_20250625_083505_utc.pt"},
                {"name": "aortic_elongation_enlargement", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_aortic_elongation_enlargement/checkpoint/checkpoint_epoch_2_20250625_130220_utc.pt"},
                {"name": "atelectasis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_atelectasis/checkpoint/checkpoint_epoch_2_20250623_183547_utc.pt"},
                {"name": "spondylitis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_spondylitis/checkpoint/checkpoint_epoch_2_20250625_223642_utc.pt"},
                {"name": "perihilar_opacity", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_perihilar_opacity/checkpoint/checkpoint_epoch_2_20250621_070111_utc.pt"},
                {"name": "venous_congestion", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_venous_congestion/checkpoint/checkpoint_epoch_2_20250622_043605_utc.pt"},
                {"name": "pleural_thickening", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pleural_thickening/checkpoint/checkpoint_epoch_2_20250630_222550_utc.pt"},
                {"name": "mediastinal_shift", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_mediastinal_shift/checkpoint/checkpoint_epoch_2_20250630_165331_utc.pt"},
                {"name": "airspace_opacity", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_airspace_opacity/checkpoint/checkpoint_epoch_2_20250623_055559_utc.pt"},
                {"name": "pneumonia", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_pneumonia/checkpoint/checkpoint_epoch_2_20250623_015630_utc.pt"},
                {"name": "fracture", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_fracture/checkpoint/checkpoint_epoch_2_20250701_035136_utc.pt"},
                {"name": "tuberculosis", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_tuberculosis/checkpoint/checkpoint_epoch_2_20250620_161905_utc.pt"},
                {"name": "edema", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_edema/checkpoint/checkpoint_epoch_2_20250621_031453_utc.pt"},
                {"name": "emphysema", "path": "/mnt/training/classifier/checkpoints/chest/intern_vit_classifier-training-on-combined_dataset_emphysema/checkpoint/checkpoint_epoch_2_20250708_171118_utc.pt"}
            ]
        },
        "head": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_head_fracture/checkpoint/checkpoint_epoch_2_20250621_235944_utc.pt"},
                {"name": "calvarial_lesions", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_head_calvarial_lesions/checkpoint/checkpoint_epoch_2_20250621_233141_utc.pt"},
                {"name": "sinusitis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_head_sinusitis/checkpoint/checkpoint_epoch_2_20250622_020400_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_head_support_devices/checkpoint/checkpoint_epoch_2_20250622_023741_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_head_no_findings/checkpoint/checkpoint_epoch_2_20250622_013013_utc.pt"}
            ]
        },
        "extremities": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_fracture/checkpoint/checkpoint_epoch_2_20250622_210917_utc.pt"},
                {"name": "dislocation", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_dislocation/checkpoint/checkpoint_epoch_2_20250622_083303_utc.pt"},
                {"name": "arthritic_changes", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_arthritic_changes/checkpoint/checkpoint_epoch_2_20250621_215137_utc.pt"},
                {"name": "joint_effusion", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_joint_effusion/checkpoint/checkpoint_epoch_2_20250623_020018_utc.pt"},
                {"name": "soft_tissue_swelling", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_soft_tissue_swelling/checkpoint/checkpoint_epoch_2_20250624_120657_utc.pt"},
                {"name": "subcutaneous_gas", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_subcutaneous_gas/checkpoint/checkpoint_epoch_2_20250624_134539_utc.pt"},
                {"name": "foreign_bodies", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_foreign_bodies/checkpoint/checkpoint_epoch_2_20250622_110557_utc.pt"},
                {"name": "bone_lesions", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_bone_lesions/checkpoint/checkpoint_epoch_2_20250622_015750_utc.pt"},
                {"name": "osteomyelitis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_osteomyelitis/checkpoint/checkpoint_epoch_2_20250624_034527_utc.pt"},
                {"name": "periosteal_reaction", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_periosteal_reaction/checkpoint/checkpoint_epoch_2_20250624_041906_utc.pt"},
                {"name": "erosive_change", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_erosive_change/checkpoint/checkpoint_epoch_2_20250622_092652_utc.pt"},
                {"name": "bony_mineralization", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_bony_mineralization/checkpoint/checkpoint_epoch_2_20250622_064207_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_support_devices/checkpoint/checkpoint_epoch_2_20250624_202211_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_extremities_no_findings/checkpoint/checkpoint_epoch_2_20250624_005855_utc.pt"}
            ]
        },
        "spine": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_fracture/checkpoint/checkpoint_epoch_2_20250622_173704_utc.pt"},
                {"name": "degenerative_changes", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_degenerative_changes/checkpoint/checkpoint_epoch_2_20250622_144953_utc.pt"},
                {"name": "spondylolisthesis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_spondylolisthesis/checkpoint/checkpoint_epoch_2_20250624_035912_utc.pt"},
                {"name": "scoliosis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_scoliosis/checkpoint/checkpoint_epoch_2_20250623_225230_utc.pt"},
                {"name": "vertebral_height_loss", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_vertebral_height_loss/checkpoint/checkpoint_epoch_2_20250624_113450_utc.pt"},
                {"name": "osteophytes", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_osteophytes/checkpoint/checkpoint_epoch_2_20250623_155718_utc.pt"},
                {"name": "bone_lesions", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_bone_lesions/checkpoint/checkpoint_epoch_2_20250621_044556_utc.pt"},
                {"name": "spondylolysis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_spondylolysis/checkpoint/checkpoint_epoch_2_20250624_051156_utc.pt"},
                {"name": "bony_mineralization", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_bony_mineralization/checkpoint/checkpoint_epoch_2_20250621_074102_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_support_devices/checkpoint/checkpoint_epoch_2_20250624_092309_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_spine_no_findings/checkpoint/checkpoint_epoch_2_20250623_064913_utc.pt"}
            ]
        },
        "pelvis": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "fracture", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_fracture/checkpoint/checkpoint_epoch_2_20250622_150257_utc.pt"},
                {"name": "hip_dislocation", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_hip_dislocation/checkpoint/checkpoint_epoch_2_20250622_154733_utc.pt"},
                {"name": "arthritis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_arthritis/checkpoint/checkpoint_epoch_2_20250622_103807_utc.pt"},
                {"name": "bone_lesions", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_bone_lesions/checkpoint/checkpoint_epoch_2_20250622_125406_utc.pt"},
                {"name": "avascular_necrosis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_avascular_necrosis/checkpoint/checkpoint_epoch_2_20250622_121837_utc.pt"},
                {"name": "asymmetry", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_asymmetry/checkpoint/checkpoint_epoch_2_20250622_113130_utc.pt"},
                {"name": "soft_tissue_swelling", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_soft_tissue_swelling/checkpoint/checkpoint_epoch_2_20250622_233325_utc.pt"},
                {"name": "subcutaneous_gas", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_subcutaneous_gas/checkpoint/checkpoint_epoch_2_20250623_002025_utc.pt"},
                {"name": "soft_tissue_calcifications", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_soft_tissue_calcifications/checkpoint/checkpoint_epoch_2_20250622_225153_utc.pt"},
                {"name": "bony_mineralization", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_bony_mineralization/checkpoint/checkpoint_epoch_2_20250622_135416_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_support_devices/checkpoint/checkpoint_epoch_2_20250623_022723_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_pelvis_no_findings/checkpoint/checkpoint_epoch_2_20250622_205836_utc.pt"}
            ]
        },
        "abdomen": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "bowel_obstruction", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_bowel_obstruction/checkpoint/checkpoint_epoch_2_20250621_091553_utc.pt"},
                {"name": "abnormal_calcifications", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_abnormal_calcifications/checkpoint/checkpoint_epoch_2_20250621_063943_utc.pt"},
                {"name": "foreign_bodies", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_foreign_bodies/checkpoint/checkpoint_epoch_2_20250621_121020_utc.pt"},
                {"name": "organomegaly", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_organomegaly/checkpoint/checkpoint_epoch_2_20250621_194341_utc.pt"},
                {"name": "abnormal_gas_patterns", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_abnormal_gas_patterns/checkpoint/checkpoint_epoch_2_20250621_082121_utc.pt"},
                {"name": "masses", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_masses/checkpoint/checkpoint_epoch_2_20250621_134520_utc.pt"},
                {"name": "ileus", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_ileus/checkpoint/checkpoint_epoch_2_20250621_130129_utc.pt"},
                {"name": "constipation", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_constipation/checkpoint/checkpoint_epoch_2_20250621_111327_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_support_devices/checkpoint/checkpoint_epoch_2_20250621_222605_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_abdomen_no_findings/checkpoint/checkpoint_epoch_2_20250621_183437_utc.pt"}
            ]
        },
        "neck": {
            "num_multi_images": 1,
            "checkpoints": [
                {"name": "listhesis", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_listhesis/checkpoint/checkpoint_epoch_2_20250622_045334_utc.pt"},
                {"name": "degenerative_changes", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_degenerative_changes/checkpoint/checkpoint_epoch_2_20250622_042041_utc.pt"},
                {"name": "calcifications", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_calcifications/checkpoint/checkpoint_epoch_2_20250622_030638_utc.pt"},
                {"name": "cervical_spine_alignment", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_cervical_spine_alignment/checkpoint/checkpoint_epoch_2_20250622_033922_utc.pt"},
                {"name": "support_devices", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_support_devices/checkpoint/checkpoint_epoch_2_20250622_062527_utc.pt"},
                {"name": "no_findings", "path": "/mnt/training/classifier/checkpoints/non-chest/cls_nonchest-training-on-combined_neck_no_findings/checkpoint/checkpoint_epoch_2_20250622_055129_utc.pt"}
            ]
        }
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
        device="cuda")

    output = classifier.predict(group="chest", dicom_files=DICOM_FILES)

    print("Predicted values:")
    print(output)
