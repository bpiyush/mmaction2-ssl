"""Converts webm files to mp4"""
import os
from os.path import join, basename, exists
import numpy as np
from glob import glob
from subprocess import call

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

    trainval_raw = read_txt(join(args.output_dir, "annotations", "ava_file_names_trainval_v2.1.txt"))
    trainval_raw = set([basename(x) for x in trainval_raw])
    webm = [x for x in trainval_raw if x.endswith(".webm")]

    for d in webm:
        source = join(args.output_dir, "videos", d)
        print(f"::::::::::::: Converting {source} to mp4")
        call(
            f"ffmpeg -fflags +genpts -i {source} -r 30 {source.replace('webm', 'mp4')}",
            shell=True,
        )
        os.remove(source)


