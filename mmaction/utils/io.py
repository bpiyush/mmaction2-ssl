"""Input output utils"""
import zipfile
import gzip
import pickle


def load_txt(filepath):
    """Reads txt file."""
    with open(filepath, "rb") as file:
        lines = file.readlines()
        lines = [line.rstrip().decode("utf-8") for line in lines]
    return lines


def save_txt(data: dict, path: str):
    """Writes data (lines) to a txt file.
    Args:
        data (dict): List of strings
        path (str): path to .txt file
    """
    assert isinstance(data, list)

    lines = "\n".join(data)
    with open(path, "w") as f:
        f.write(str(lines))


def unzip_file(file, dir):
    """Unzips a given file in given dir."""
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dir)


def load_pkl(file: str, encoding="latin1"):
    """Loads a pickle file."""
    with open(file, "rb") as f:
        data = pickle.load(f, encoding=encoding)
    return data