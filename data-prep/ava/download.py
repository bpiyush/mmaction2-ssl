"""
Setup script for AVA dataset

Example:
(conda-env) $ python download.py -o /ssd/pbagad/datasets/AVA/
"""

import time
from os import makedirs
from os.path import join, exists, basename, isdir
from tqdm import tqdm
import requests
import wget

from mmaction.utils.io import read_txt, unzip_file


TRAIN_FILE_LIST_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
TRAIN_FILE_TEMPLATE = "https://s3.amazonaws.com/ava-dataset/trainval/{}"

TEST_FILE_LIST_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_test_v2.1.txt"
TEST_FILE_TEMPLATE = "https://s3.amazonaws.com/ava-dataset/test/{}"

ANNOTATION_FILE_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip"


def download_file(url, save_path, desc="Downloading sample file", overwrite=False, block_size=102400):
    """
    Downloads file at given URL with a progress bar.

    Inspired: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests

    Args:
        block_size (int): defaults to 100 Kibibyte (0.1 MB)
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))

    if not exists(save_path) or overwrite:
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=desc)

        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"ERROR, something went wrong while downloading {url}")
    else:
        print(f"WARNING: File already exists at {save_path} and overwrite=False.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir",
        help="Parent dataset folder",
        required=True,
    )
    parser.add_argument(
        "-f", "--force_overwrite",
        help="Overwrite existing dataset files by downloading",
        action="store_true",
    )
    args = parser.parse_args()

    start = time.time()

    assert isdir(args.output_dir), \
        f"Given dataset output folder does not exist at {args.output_dir}"

    # step 0: create dataset folders
    raw_dir = join(args.output_dir, "raw")
    makedirs(raw_dir, exist_ok=True)
    videos_dir = join(raw_dir, "videos")
    makedirs(videos_dir, exist_ok=True)

    annot_dir = join(args.output_dir, "annotations")
    makedirs(annot_dir, exist_ok=True)

    meta_dir = join(args.output_dir, "meta")
    makedirs(meta_dir, exist_ok=True)

    # step 1: download train and test file lists
    train_list_path = join(meta_dir, basename(TRAIN_FILE_LIST_URL))
    if not exists(train_list_path) or args.force_overwrite:
        print("Downloading train files list: ")
        train_list_path = wget.download(url=TRAIN_FILE_LIST_URL, out=meta_dir)
        print("\n")
    else:
        print(f"Train file already exists at {train_list_path}")

    test_list_path = join(meta_dir, basename(TEST_FILE_LIST_URL))
    if not exists(test_list_path) or args.force_overwrite:
        print("Downloading test files list: ")
        test_list_path = wget.download(url=TEST_FILE_LIST_URL, out=meta_dir)
        print("\n")
    else:
        print(f"Test file already exists at {test_list_path}")

    # step 2: download annotation files
    annot_file = join(annot_dir, basename(ANNOTATION_FILE_URL))
    if not exists(annot_file) or args.force_overwrite:
        download_file(ANNOTATION_FILE_URL, annot_file, desc=f"Downloading annotation files")
        unzip_file(annot_file, annot_dir)
    else:
        print(f"Annotation files already exist at {annot_dir}")

    # step 3: download train and test files
    debug = False
    train_files = read_txt(train_list_path)
    print(f":::: Downloading {len(train_files)} train files.")
    for i, f in enumerate(train_files):
        url = TRAIN_FILE_TEMPLATE.format(f)
        save_path = join(videos_dir, f)
        download_file(
            url, save_path,
            desc=f"Downloading file [{i + 1}/{len(train_files)}]",
            overwrite=args.force_overwrite,
        )
        if debug:
            break

    test_files = read_txt(test_list_path)
    print(f":::: Downloading {len(test_files)} test files.")
    for i, f in enumerate(test_files):
        url = TRAIN_FILE_TEMPLATE.format(f)
        save_path = join(videos_dir, f)
        download_file(
            url, save_path,
            desc=f"Downloading file [{i + 1}/{len(test_files)}]",
            overwrite=args.force_overwrite,
        )
        if debug:
            break
    
    end = time.time()
    print(f"Completed download for AVA dataset in {(end - start) / 60.0} mins.")
