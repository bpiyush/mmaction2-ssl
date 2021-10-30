"""Input output utils"""
import zipfile


def read_txt(filepath):
    """Reads txt file."""
    with open(filepath, "rb") as file:
        lines = file.readlines()
        lines = [line.rstrip().decode("utf-8") for line in lines]
    return lines


def unzip_file(file, dir):
    """Unzips a given file in given dir."""
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dir)
