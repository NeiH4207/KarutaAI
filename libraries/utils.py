import os


def gather_file(path, ext='.wav'):
    all_files = os.listdir(path)
    return [_file for _file in all_files if _file.endswith(ext)]
