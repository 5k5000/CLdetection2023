# !/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import (keypoint_auc, keypoint_epe, keypoint_nme,
                          keypoint_pck_accuracy)
import torch

@METRICS.register_module()
class CephalometricMetric(BaseMetric):
    """Cephalometric evaluation metric.

    Calculate the Mean Radius Error of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 target_class_id = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.target_class_id = target_class_id      # 只计算某一个类的指标

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # spacing
            spacing = data_sample['spacing']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)
            # print(mask.shape)       # [1,38]

            if self.target_class_id is not None:
                pred_coords = pred_coords[:, self.target_class_id-1, :]
                gt_coords = gt_coords[:, self.target_class_id-1, :]
                mask = mask[:, self.target_class_id-1]
                if type(pred_coords) == torch.Tensor:
                    pred_coords = pred_coords.unsqueeze(1)
                    gt_coords = gt_coords.unsqueeze(1)
                    mask = mask.unsqueeze(1)
                else:   # numpy
                    pred_coords = pred_coords[:, np.newaxis]
                    gt_coords = gt_coords[:, np.newaxis]
                    mask = mask[:, np.newaxis]
                # 经过以上步骤后尺寸应该是：[1, 1, D] 和 [1, 1, 1]

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'spacing': spacing
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])
        # spacing: [N, 1]
        spacing = np.asarray([[result['spacing']] for result in results])

        logger.info(f'Evaluating {self.__class__.__name__}...')
        # pred_coords = np.round(pred_coords)
        # calculate the prediction keypoints error
        n_kpt = np.prod(np.shape(gt_coords)[:-1])
        each_kpt_error = np.sqrt(np.sum(np.square(pred_coords - gt_coords), axis=2)) * spacing

        # the mean radial error metric
        mre = np.sum(each_kpt_error) / n_kpt
        mre_std = np.std(each_kpt_error)

        # the success detection rate metric
        sdr2_0 = np.sum(each_kpt_error <= 2.0) / n_kpt * 100
        sdr2_5 = np.sum(each_kpt_error <= 2.5) / n_kpt * 100
        sdr3_0 = np.sum(each_kpt_error <= 3.0) / n_kpt * 100
        sdr4_0 = np.sum(each_kpt_error <= 4.0) / n_kpt * 100

        metrics = dict()
        metrics['MRE'] = mre
        metrics['SDR 2.0mm'] = sdr2_0
        metrics['SDR 2.5mm'] = sdr2_5
        metrics['SDR 3.0mm'] = sdr3_0
        metrics['SDR 4.0mm'] = sdr4_0

        print('=> {:<24} :  {} = {:0.3f} ± {:0.3f} mm'.format('Mean Radial Error', 'MRE', mre, mre_std))
        print('=> {:<24} :  SDR 2.0mm = {:0.3f}% | SDR 2.5mm = {:0.3f}% | SDR 3mm = {:0.3f}% | SDR 4mm = {:0.3f}%'
              .format('Success Detection Rate', sdr2_0, sdr2_5, sdr3_0, sdr4_0))

        return metrics






