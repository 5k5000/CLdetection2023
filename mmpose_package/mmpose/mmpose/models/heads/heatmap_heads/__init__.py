# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead,HeatmapHead_withSigmoid
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .srpose_head import SRPoseHead
__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead','HeatmapHead_withSigmoid','SRPoseHead'
]
