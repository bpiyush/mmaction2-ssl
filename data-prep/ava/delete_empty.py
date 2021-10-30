"""Deletes empty test files."""
import os
from os.path import join, basename, exists
import numpy as np
from glob import glob

from mmaction.utils.io import read_txt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir",
        help="Parent dataset folder",
        required=True,
    )
    args = parser.parse_args()

    test_files = read_txt(join(args.output_dir, "annotations", "ava_file_names_test_v2.1.txt"))
    test_files = [join(args.output_dir, "videos", x) for x in test_files]
    test_ids = [basename(x) for x in test_files]

    exts = [basename(x).split(".")[-1] for x in test_files]
    exts = np.unique(exts)

    for f in test_files:
        if exists(f):
            os.remove(f)
