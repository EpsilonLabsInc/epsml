import argparse
import os
import pathlib

import torch


def strip_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = {"model_state_dict": checkpoint["model_state_dict"]}

    directory, filename = os.path.split(checkpoint_path)
    new_checkpoint_path = os.path.join(directory, "production_" + filename)

    torch.save(new_checkpoint, new_checkpoint_path)


def main(args):
    print(f"Scanning {args.checkpoints_dir} for the .pt files")
    checkpoints_dir = pathlib.Path(args.checkpoints_dir)
    checkpoints = list(checkpoints_dir.rglob(args.checkpoint_name_filter))
    print(f"Found {len(checkpoints)} matching checkpoints")

    for index, checkpoint in enumerate(checkpoints):
        print("")
        print(f"{index + 1}/{len(checkpoints)} Stripping checkpoint {checkpoint}")
        print("")

        strip_checkpoint(checkpoint)


if __name__ == "__main__":
    CHECKPOINTS_DIR = "/mnt/training/classifier/checkpoints"
    CHECKPOINT_NAME_FILTER = "checkpoint_epoch_2_*.pt"

    args = argparse.Namespace(checkpoints_dir=CHECKPOINTS_DIR,
                              checkpoint_name_filter=CHECKPOINT_NAME_FILTER)

    main(args)
