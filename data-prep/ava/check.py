"""Checks downloaded dataset files."""
from os.path import join, basename
import numpy as np

from mmaction.utils.io import read_txt

# TODO: fill this later

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir",
        help="Parent dataset folder",
        required=True,
    )
    args = parser.parse_args()


    trainval_files = read_txt(join(args.output_dir, "meta", "ava_file_names_trainval_v2.1.txt"))
    trainval_files = [join(args.output_dir, "raw/videos", x) for x in trainval_files]

    exts = [basename(x).split(".")[-1] for x in trainval_files]
    exts = np.unique(exts)

