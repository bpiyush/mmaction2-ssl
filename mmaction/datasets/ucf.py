#!/usr/bin/env python3

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)
from .pipelines import Compose
from .base import BaseDataset
from .builder import DATASETS
from ..utils import load_txt


@DATASETS.register_module()
class UCFDataset(BaseDataset):
    """UCF Dataset (UCF101-24) for spatial temporal detection.

    Based on official UCF dataset and unofficial annotations.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> pyannot.pkl
        label_file -> UCF101v2-GT.pkl
        split_file -> {train,valid}_seed_0.txt

    Args:
        ann_file (str): Path to the annotation file.
        split_file (str): Path to the split (train/valid) file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """
    _FPS = 30

    def __init__(self,
                 ann_file,
                 split_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=24,
                 start_index=1,
                 modality='RGB',
                 proposal_file=None,
                 filename_tmpl='img_{:05}.jpg',
                ):
        self.split_file = split_file
        self.proposal_file = proposal_file
        self.filename_tmpl = filename_tmpl

        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            modality=modality,
            num_classes=num_classes,
            multi_class=multi_class,
            start_index=start_index,
        )

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None

    def parse_img_record(self, img_records):
        """Merge image records of the same entity at the same time.

        Args:
            img_records (list[dict]): List of img_records (lines in AVA
                annotations).

        Returns:
            tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids
        """
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            num_selected_records = len(selected_records)
            img_records = [
                x for x in img_records if
                not np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        data = mmcv.load(self.ann_file)
        videos = load_txt(self.split_file)
        metadata = mmcv.load(self.ann_file.replace("pyannot.pkl", "UCF101v2-GT.pkl"), encoding="latin1")

        records_dict_by_img = defaultdict(list)
        video_infos = []

        # iterate over each video
        for video_id in tqdm(videos, "Reading video annotations for UCF101_24"):

            video_ann = data[video_id]["annotations"]
            timestamp_start = 1
            timestamp_end = data[video_id]["numf"]

            resolution = metadata["resolution"][video_id]
            bbox_div_factor = np.array(
                [resolution[1], resolution[0], resolution[1], resolution[0]]
            )

            # iterate over each tubelet annotated in this video
            for entity_id, ann in enumerate(video_ann):
                tube_start = ann["sf"]
                tube_end = ann["ef"]
                tube_frames = np.arange(tube_start + 1, tube_end + 1, 1)
                tube_length = len(tube_frames)
                num_boxes = ann["boxes"].shape[0]
                tube_label = ann["label"]

                assert tube_length == num_boxes, \
                    f"Number of frames in tube ({tube_length}) != Annotated boxes ({num_boxes})"

                # iterate over each annotated frame in the tubelet
                for j, timestamp in enumerate(tube_frames):
                    img_key = f"{video_id},{timestamp}"

                    # entity_box = np.array(list(map(float, line_split[2:6])))
                    entity_box = (ann["boxes"][j]).astype(np.float32)
                    entity_box = np.divide(entity_box, bbox_div_factor)

                    # entity_id = int(line_split[7])
                    shot_info = (0, (timestamp_end - timestamp_start) * self._FPS)

                    video_info = dict(
                        video_id=video_id,
                        timestamp=timestamp,
                        entity_box=entity_box,
                        label=tube_label,
                        entity_id=entity_id,
                        shot_info=shot_info,
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        )
                    records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            timestamp_start = records_dict_by_img[img_key][0]["timestamp_start"]
            timestamp_end = records_dict_by_img[img_key][0]["timestamp_end"]
            bboxes, labels, entity_ids = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                timestamp_start=int(timestamp_start),
                timestamp_end=int(timestamp_end),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann)
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        # results['timestamp_start'] = self.timestamp_start
        # results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        # results['timestamp_start'] = self.timestamp_start
        # results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        # Follow the mmdet variable naming style.
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)


if __name__ == "__main__":
    pipeline = [
        dict(dict(type='SampleAVAFrames', clip_len=32, frame_interval=2))
    ]

    data_root = osp.normpath(osp.join(osp.dirname(__file__), '../../data', 'ucf101_24'))

    data_prefix = osp.join(data_root, "rgb-images")
    ann_file = osp.join(data_root, "pyannot.pkl")
    split_file = osp.join(data_root, "valid_seed_0.txt")

    dataset = UCFDataset(ann_file, split_file, pipeline, data_prefix=data_prefix)
    X = dataset[0]

    import ipdb; ipdb.set_trace()