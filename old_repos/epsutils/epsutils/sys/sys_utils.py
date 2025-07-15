import os


def apply_path_substitutions(path, path_substitutions):
    for old, new in path_substitutions.items():
        path = path.replace(old, new)

    return path


def get_containing_dir(path):
    return os.path.basename(os.path.normpath(path))


def compute_dir_depth(dir1, dir2):
    # Normalize paths to avoid issues with different slash formats.
    dir1 = os.path.normpath(dir1)
    dir2 = os.path.normpath(dir2)

    # Check if one dir is a subdirectory of the other.
    if not (dir1.startswith(dir2) or dir2.startswith(dir1)):
        raise ValueError("Neither dir is a subdirectory of the other")

    # Compute depth difference.
    depth1 = dir1.count(os.sep)
    depth2 = dir2.count(os.sep)

    return abs(depth1 - depth2)


def add_suffix_to_file_path(file_path, suffix):
    dir_name = os.path.dirname(file_path)
    base_name, extension = os.path.splitext(os.path.basename(file_path))
    new_file_name = f"{base_name}{suffix}{extension}"
    return os.path.join(dir_name, new_file_name)
