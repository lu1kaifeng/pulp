import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def from_root_dir(relative_path: str):
    return os.path.join(ROOT_DIR, relative_path)
