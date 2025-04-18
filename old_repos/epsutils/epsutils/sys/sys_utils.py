
def apply_path_substitutions(path, path_substitutions):
    for old, new in path_substitutions.items():
        path = path.replace(old, new)

    return path
