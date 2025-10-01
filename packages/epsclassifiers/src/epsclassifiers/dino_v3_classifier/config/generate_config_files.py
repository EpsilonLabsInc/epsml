import argparse
import copy
import os
import sys
import yaml

from epsutils.labels.labels_by_body_part import LABELS_BY_BODY_PART


CKPT_VARIANTS = {
    # name -> teacher checkpoint path
    "005_015": "/mnt/all-data/files/output_vitl_xray_20_ep_005_015/eval/training_39279/teacher_checkpoint.pth",
    "005_032": "/mnt/all-data/files/output_vitl_xray_20_ep_005_032/eval/training_39279/teacher_checkpoint.pth",
    "005_015_032": "/mnt/all-data/files/output_vitl_xray_20_ep_005_015_032/eval/training_39279/teacher_checkpoint.pth",
    "005_015_032_768": "/mnt/all-data/files/output_vitl_xray_20_ep_005_015_032_768/eval/training_78559/teacher_checkpoint.pth",
}


def _load_base_config(template_path: str) -> dict:
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def _swap_512_with_org_size(cfg: dict) -> None:
    """Replace '/png/512x512/' with '/png/org-size/' in base path substitutions."""
    base_subs = cfg.get("paths", {}).get("base_path_substitutions", {})
    if not isinstance(base_subs, dict):
        return
    updated = {}
    for k, v in base_subs.items():
        if isinstance(v, str):
            updated[k] = v.replace("/png/512x512/", "/png/org-size/")
        else:
            updated[k] = v
    cfg["paths"]["base_path_substitutions"] = updated


"""
Special-case epoch overrides for specific labels using exact names from
epsutils.labels.labels_by_body_part.LABELS_BY_BODY_PART.
"""
SPECIAL_TWO_EPOCHS_EXACT = {
    "Foot": ["Degenerative changes"],
    "Leg": ["Joint effusion"],
    "Chest": ["Mass/nodule"],
    "Hand": ["Degenerative changes"],
    "C-spine": ["Intervertebral disc narrowing"],
    "T-spine": ["Scoliosis"],
}


def generate_for_body_part(body_part: str, run_name: str, output_root: str,
                           chest_config_template: str, non_chest_config_template: str) -> None:
    # Validate body part (case-insensitive match to known keys)
    valid_keys = list(LABELS_BY_BODY_PART.keys())
    normalized_map = {k.lower(): k for k in valid_keys}
    if body_part.lower() not in normalized_map:
        print(f"Warning: Body part '{body_part}' not in known set: {', '.join(valid_keys)}")
        # Proceeding but labels may be missing; bail early if unknown
        return
    canon_body_part = normalized_map[body_part.lower()]

    # Load template based on body part
    config_template = chest_config_template if canon_body_part == "Chest" else non_chest_config_template
    base_cfg = _load_base_config(config_template)

    # Get labels
    labels = LABELS_BY_BODY_PART[canon_body_part]

    # Generate configs
    for ckpt_name, ckpt_path in CKPT_VARIANTS.items():
        for label in labels:
            formatted_body_part = canon_body_part.lower().replace(' ', '_').replace("/", "_")
            formatted_label = label.lower().replace(' ', '_').replace("/", "_")

            cfg = copy.deepcopy(base_cfg)

            cfg["general"]["dataset_name"] = f"{formatted_body_part}_{formatted_label}"
            # Name runs after the checkpoint variant to compare initializations in W&B
            cfg["general"]["run_name"] = ckpt_name
            cfg["general"]["notes"] = ""
            cfg["general"]["custom_labels"] = [label]
            cfg["general"]["body_part"] = canon_body_part

            cfg.setdefault("paths", {})
            cfg["paths"]["dino_v3_checkpoint_path"] = ckpt_path

            if ckpt_name.endswith("_768"):
                _swap_512_with_org_size(cfg)
                if "general" in cfg:
                    cfg["general"]["dino_v3_img_size"] = 768
                # Reduce batch sizes for 768 variant to avoid OOM
                cfg.setdefault("training", {})
                cfg["training"]["training_batch_size"] = 128
                cfg["training"]["validation_batch_size"] = 128

            # Set num_epochs: 2 for specific label/body-part combos, otherwise 4
            cfg.setdefault("training", {})
            two_epoch_targets = set(SPECIAL_TWO_EPOCHS_EXACT.get(canon_body_part, []))
            epochs = 2 if label in two_epoch_targets else 4
            cfg["training"]["num_epochs"] = epochs

            output_dir = os.path.join(output_root, ckpt_name, formatted_body_part)
            os.makedirs(output_dir, exist_ok=True)

            out_file = os.path.abspath(os.path.join(output_dir, f"{formatted_body_part}_{formatted_label}_config.yaml"))
            print(f"Saving config file {out_file}")
            with open(out_file, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)


def main(argv=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate DINOv3 classifier configs by body part and checkpoint variant.")
    parser.add_argument("--body_parts", type=str, default="T-spine",
                        help="Comma-separated list of body parts, or 'all' to generate for all known body parts.")
    parser.add_argument("--run_name", type=str, default="Training with attention pooling",
                        help="Run name to set in generated configs.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "generated/with_attention_pooling"),
                        help="Output directory for generated configs.")
    parser.add_argument("--chest_config_template", type=str,
                        default=os.path.join(BASE_DIR, "template/with_attention_pooling/chest_with_attention_pooling_config_template.yaml"),
                        help="Path to chest template YAML.")
    parser.add_argument("--non_chest_config_template", type=str,
                        default=os.path.join(BASE_DIR, "template/with_attention_pooling/non_chest_with_attention_pooling_config_template.yaml"),
                        help="Path to non-chest template YAML.")

    args = parser.parse_args(argv)

    if args.body_parts.strip().lower() == "all":
        body_parts = list(LABELS_BY_BODY_PART.keys())
    else:
        body_parts = [bp.strip() for bp in args.body_parts.split(",") if bp.strip()]

    for bp in body_parts:
        generate_for_body_part(bp, args.run_name, args.output_dir,
                               args.chest_config_template, args.non_chest_config_template)

    print("Generation of config files completed successfully.")


if __name__ == "__main__":
    sys.exit(main())
