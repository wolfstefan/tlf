# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS
from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy, multi_top_k_accuracy)

import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset

@DATASETS.register_module()
class YouTubeDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().rsplit(' ', 1)
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1].split(",")
                    label = list(map(int, label))
                else:
                    filename, label = line_split[0], line_split[1]
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 label_mapping=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            label_mapping (dict): In the case of multi-class multi-label training, a dictionary mapping each coarse label to a fine one.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'multi_top_k_accuracy', 'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        # NOTE: In case the TSNSC head is used, the results will not have the shape [num_videos, num_classes],
        # but [num_videos, num_frames, num_classes]. To make sure the evaluation still works, the tensor needs
        # to be transformed first into the following shape: [num_videos * num_frames, num_classes].
        if (np.array(results).ndim == 3):
            results = np.array(results)
            rs = results.shape
            results = results.reshape(rs[0] * rs[1], rs[2])

            # Each element in the gt_labels array needs to be repeated rs[1] (num_frames) times
            # to fit the reshape of the results array.
            gt_labels = np.array(gt_labels)
            gt_labels = np.repeat(gt_labels, repeats=rs[1])

        if label_mapping is not None:
            for r in range(len(results)):
                # Option 1: Add abs(x) to everything, where x is the minimum score, so everything is positive or 0
                    # minval = abs(min(results[r])) + 1
                    # results[r] = [i + minval for i in results[r]]
                # Option 2: Sigmoid the scores
                sigm = lambda inval: 1 / (1 + np.exp(-inval))
                results[r] = np.vectorize(sigm)(results[r])

                for coarse_idx in label_mapping.keys():
                    for fine_idx in label_mapping[coarse_idx]:
                        results[r][fine_idx] *= results[r][coarse_idx]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'multi_top_k_accuracy':
                topk = metric_options.setdefault('multi_top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_accs = multi_top_k_accuracy(results, gt_labels, [(0, 196), (196, 245)], topk)
                for idx, labelacc in enumerate(top_k_accs):
                    log_msg = []
                    for k, acc in zip(topk, labelacc):
                        eval_results[f'top{k}_acc'] = acc
                        log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                    log_msg = 'label #{}: '.format(idx) + ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results
