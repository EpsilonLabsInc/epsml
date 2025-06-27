import pandas as pd
import yaml
import os

# csv from sheet of https://docs.google.com/spreadsheets/d/13tes-CRhP1tp2iwnBn5_wN2cv7D9lBFJRWCRUJwfTAk/edit?gid=0#gid=0
info_df = pd.read_csv("./non-chest-training.csv")
# only keep the columns we need based on number of training examples
info_df = info_df[info_df['use'] == 1]


paths_section = {
    'intern_vl_checkpoint_dir': "/mnt/training/internvl3_chimera_20250609_233409_1e-5_epsilon_all_0608/checkpoint-24934/",
    'train_file': "/home/eric/projects/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_train.csv",
    'validation_file': "/home/eric/projects/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_val.csv",
    'test_file': "/home/eric/projects/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_test.csv",
    'base_path_substitutions': {
        "gradient/22JUL2024": None,
        "gradient/20DEC2024": None,
        "gradient/09JAN2025": None,
        "gradient/16AUG2024": "/mnt/sfs-gradient-nochest/16AUG2024/",
        "gradient/13JAN2025": "/mnt/sfs-gradient-nochest/13JAN2025/deid",
        "segmed/batch1": "/mnt/sfs-segmed-1",
        "segmed/batch2": "/mnt/sfs-segmed-2",
        "segmed/batch3": "/mnt/sfs-segmed-34/segmed_3",
        "segmed/batch4": "/mnt/sfs-segmed-34/segmed_4",
        "simonmed": "/mnt/sfs-simonmed"
    },
    'output_dir': "./output"
}

training_section = {
    'perform_intra_epoch_validation': True,
    'intra_epoch_validation_step': 3000,
    'send_wandb_notification': False,
    'device': "cuda",
    'device_ids': None,
    'num_training_workers_per_gpu': 32,
    'num_validation_workers_per_gpu': 32,
    'learning_rate': 0.01,
    'warmup_ratio': 0.05,
    'num_epochs': 10,
    'training_batch_size': 128,
    'validation_batch_size': 32,
    'min_allowed_batch_size': 2,
    'multi_image_input': False
}


def generate_yaml(fixed_num_aug=0):

    for _, row in info_df.iterrows():
        part = row['Part']
        label = row['Label']
        num_aug = fixed_num_aug if fixed_num_aug else int(row['Num augmentations'])

        # lowercase/underscore variants
        part_l = part.lower()
        label_l = label.lower().replace(' ', '_')
        dataset_name = f"combined_{part_l}_{label_l}"
        out_dir = os.path.join("/home/eric/projects/epsclassifiers/epsclassifiers/intern_vit_classifier/config/non-chest/", part_l)
        filename = f"combined_dataset_{part_l}_{label_l}.yaml"
        full_path = os.path.join(out_dir, filename)

        yaml_dict = {
            'general': {
                'model_name': "cls_nonchest",
                'dataset_name': dataset_name,
                'run_name': f"{part_l} {label_l} - release run with {num_aug}x pos data aug",
                'notes': f"Balanced labels with {num_aug}x pos data augmentation",
                'custom_labels': [label],
                'body_part': part,
                'treat_uncertain_as_positive': True,
                'perform_label_balancing': True,
                'num_data_augmentations': num_aug,
                'save_full_model': False
            },
            'paths': paths_section,
            'training': training_section
        }

        # print(yaml_dict)

        # make sure the part‐specific folder exists
        os.makedirs(out_dir, exist_ok=True)
        # write YAML
        with open(full_path, 'w') as f:
            yaml.dump(yaml_dict, f, sort_keys=False)

        print(f"Wrote {full_path}")


if __name__ == "__main__":
    fixed_num_aug = 0
    generate_yaml(fixed_num_aug=fixed_num_aug)