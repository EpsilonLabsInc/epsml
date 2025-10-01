import argparse
import json

from tqdm import tqdm

from intern_vit_composite_binary_classifier import InternVitCompositeBinaryClassifier


def main(args):
    # Create classifier.
    classifier = InternVitCompositeBinaryClassifier(
        grouped_binary_classifier_checkpoints=args.grouped_binary_classifier_checkpoints,
        intern_vl_checkpoint_dir=args.intern_vl_checkpoint_dir,
        intern_vit_output_dim=3200,
        device="cuda")

    # Get number of lines.
    with open(args.input_jsonl_file, "r") as file:
        total_lines = sum(1 for _ in file)

    # Run inference.
    results = []
    with open(args.input_jsonl_file, "r") as file:
        for line in tqdm(file, total=total_lines, desc="Running inference"):
            data = json.loads(line)

            image_paths = data["image"]
            body_part =  data["labels"][0]["body_part"].lower()

            if body_part == "chest":
                probs = None
            elif body_part not in args.grouped_binary_classifier_checkpoints:
                print(f"WARNING: Unknown body part '{body_part}'")
                probs = None
            else:
                probs = classifier.predict(group=body_part, dicom_files=image_paths)
                probs = {item["name"]: item["probs"].item() for item in probs}

            data["probs"] = probs
            results.append(data)

    # Save results.
    print(f"Saving results to {args.output_jsonl_file}")
    with open(args.output_jsonl_file, "w") as file:
        for item in results:
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    INPUT_JSONL_FILE = "/home/andrej/sampled_1000_nolabels.jsonl"
    OUTPUT_JSONL_FILE = "/home/andrej/sampled_1000_nolabels_with_probs.jsonl"
    INTERN_VL_CHECKPOINT_DIR = "/mnt/training/internvl3_chimera_20250609_233409_1e-5_epsilon_all_0608/checkpoint-24934/"
    GROUPED_BINARY_CLASSIFIER_CHECKPOINTS = {
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

    args = argparse.Namespace(input_jsonl_file=INPUT_JSONL_FILE,
                              output_jsonl_file=OUTPUT_JSONL_FILE,
                              intern_vl_checkpoint_dir=INTERN_VL_CHECKPOINT_DIR,
                              grouped_binary_classifier_checkpoints=GROUPED_BINARY_CLASSIFIER_CHECKPOINTS)

    main(args)
