# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss,FocalHeatmapLoss_5k,JointsMSELoss,KeypointL1Loss)
from .loss_wrappers import CombinedLoss, MultipleLossWrapper,MultipleLossWrapper_5k
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
    'KeypointMSELoss','KeypointL1Loss','KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss','FocalHeatmapLoss_5k',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss','JointsMSELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss','MultipleLossWrapper_5k'
]
