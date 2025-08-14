import argparse

from epsutils.tar import tar_utils


def main(args):
    tar_utils.extract_tar_archives(root_dir=args.dataset_root_dir,
                                   max_workers=args.num_workers,
                                   delete_after_extraction=args.delete_tar_archives_after_extraction)


if __name__ == "__main__":
    DATASET_ROOT_DIR = "/mnt/all-data/sfs-gradient-new/01JUL2025"
    NUM_WORKERS = 1  # > 1 is unsafe because multiple TARs can contain overlapping paths.
    DELETE_TAR_ARCHIVES_AFTER_EXTRACTION = True

    args = argparse.Namespace(dataset_root_dir=DATASET_ROOT_DIR,
                              num_workers=NUM_WORKERS,
                              delete_tar_archives_after_extraction=DELETE_TAR_ARCHIVES_AFTER_EXTRACTION)

    main(args)
