# Copyright (c) OpenMMLab. All rights reserved.

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


import copy
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@DATASETS.register_module()
class CephalometricDataset(BaseCocoStyleDataset):
    """Cephalometric dataset for skull landmarks detection.

    "MICCAI 2023 Challenge", MICCAI'2023.
    More details can be found in the https://cl-detection2023.grand-challenge.org.

    Cephalometric keypoints::

        0: '',
        1: '',
        2: '',
        3: '',
        4: '',
        5: '',
        6: '',
        7: '',
        8: '',
        9: '',
        10: '',
        11: '',
        12: '',
        13: '',
        14: '',
        15: '',
        16: '',
        17: '',
        18: '',
        19: '',
        20: '',
        21: '',
        22: '',
        23: '',
        24: '',
        25: '',
        26: '',
        27: '',
        28: '',
        29: '',
        30: '',
        31: '',
        32: '',
        33: '',
        34: '',
        35: '',
        36: '',
        37: '',

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/cephalometric.py')
    
    
    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 use_outer_bbox: bool = False, # wuqian加的 用来控制是否使用所有keypoint的外包矩形来作为bbox，这里的外包矩形会扩大一些
                 ):
        self.use_outer_bbox = use_outer_bbox    # 是否使用外包矩形作为bbox
        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse COCO annotation of an instance with SPACING value.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        # 计算所有keypoints的外包矩形
        outer_bbox_x1 = np.min(keypoints[0,:,0])
        outer_bbox_x2 = np.max(keypoints[0,:,0])
        outer_bbox_y1 = np.min(keypoints[0,:,1])
        outer_bbox_y2 = np.max(keypoints[0,:,1])
        
        outer_bbox = np.array([outer_bbox_x1-20, outer_bbox_y1-20, outer_bbox_x2+20, outer_bbox_y2+20], dtype=np.float32).reshape(1, 4)
        
        
        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': outer_bbox if self.use_outer_bbox else bbox,    # wuqian修改了这里，如果use_outer_bbox为True，则使用外包矩形作为bbox
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
            'spacing': img['spacing'],
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info