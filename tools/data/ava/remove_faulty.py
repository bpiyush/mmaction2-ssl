"""Removes empty videos from train/val"""
import os
from os.path import join
import pandas as pd

from mmaction.utils.io import read_txt


DATA_DIR = "../../../data/ava/"
ANNO_DIR = join(DATA_DIR, "annotations")

rm_ids = read_txt(join(ANNO_DIR, "remove.txt"))

# remove from train
trains = pd.read_csv(join(ANNO_DIR, "ava_train_v2.1.csv"), header=None)
indices = trains[0].isin(rm_ids)
trains_trimmed = trains[~indices]
trains_trimmed.to_csv(join(ANNO_DIR, "ava_train_trimmed_v2.1.csv"), header=False, index=False)

# remove from valid
valids = pd.read_csv(join(ANNO_DIR, "ava_val_v2.1.csv"), header=None)
indices = valids[0].isin(rm_ids)
valids_trimmed = valids[~indices]
valids_trimmed.to_csv(join(ANNO_DIR, "ava_val_trimmed_v2.1.csv"), header=False, index=False)
