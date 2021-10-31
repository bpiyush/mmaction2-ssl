"""Extracts RGB frames using ffmpeg"""
import os
from os.path import join
from glob import glob
from subprocess import call

DATA_DIR = "../../../data/ava/"

webm_ids = [
    "_dBTTYDRdRQ",
    "2FIHxnZKg6A",
    "8nO5FFbIAog",
    "c9pEMjPT16M",
    "Gvp-cj3bmIY",
    "jE0S8gYWftE",
    "QCLQYnt3aMo",
    "Riu4ZKk4YdQ",
    "uNT6HrrnqPU",
    "xeGWXqSvC-8",
]

for vid in webm_ids:
    video = join(DATA_DIR, "videos_15min", f"{vid}.mp4")
    out_video_dir = join(DATA_DIR, "rawframes", vid)
    os.makedirs(out_video_dir, exist_ok=True)
    out_name = f"{out_video_dir}/img_%05d.jpg"

    command = f"ffmpeg -i {video} -r 30 -q:v 1 {out_name}"
    print(command)
    call(command, shell=True)
