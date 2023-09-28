# Fivethousand reimplemented SRPose Head        # ACMMM 2023 paper referred
import pdb
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer)
from mmengine.model import constant_init,normal_init
from typing import Optional, Sequence, Tuple, Union
# from mmpose.core.evaluation import pose_pck_accuracy
# from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from mmpose.registry import KEYPOINT_CODECS, MODELS
import torch.nn.functional as F
from ..base_head import BaseHead
from torch import Tensor, nn
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from mmpose.models.utils.tta import flip_heatmaps
from mmengine.structures import PixelData


class ConvergeHead(nn.Module):
    def __init__(self, in_dim, up_ratio, kernel_size, padding, num_joints):
        super().__init__()
        self.in_dim = in_dim     
        self.up_ratio = up_ratio
        self.num_joints = num_joints

        self.conv = nn.Conv2d(in_dim*num_joints, (up_ratio**2)*num_joints, 
            kernel_size, 1, padding, 1, num_joints)
        self.apply(self._init_weights)

    def forward(self, x):
        hp = self.conv(x)
        hp = F.pixel_shuffle(hp, self.up_ratio)
        return hp

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
    

@MODELS.register_module()
class SRPoseHead(BaseHead):
    """ SRPose

    Args:
        in_channels (Sequence[int]): Numbers of input channels
        out_channels (Sequence[int]): Numbers of output channels
        num_joints (int): Number of joints.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        upsample_log (Sequence[int]): Numbers of upsample log
        per_emb_nums (Sequence[int]): Numbers of each keypoint embedding channel
        supervises (Sequence[bool]): Supervise or not
        detector_type (string): Default: 'TopDownHigher'.
    """

    def __init__(self,
                 in_channels=[2048, 1024, 512, 256],
                 out_channels=[256,128,64,32],
                 num_joints=38,     # 类别数
                 extra=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample_log=[3, 2, 1, 2],
                 per_emb_nums=[16, 8, 4, 4],
                init_cfg = None,
                supervises=[True, True, True, True],
                decoder: OptConfigType = None,  
                 specified_target_class = None,
                 loss: ConfigType = None,
                 detector_type = None
                 ):
    
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.loss_module = build_loss(loss)
        self.upsample_log = upsample_log
        self.supervises = supervises
        self.detector_type = detector_type

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        self.lr_head = nn.Sequential(
            nn.Conv2d(in_channels[0], num_joints, 1),
            nn.ReLU()
        )
        self.lr_fuse = nn.Sequential(
            nn.Conv2d(num_joints, out_channels[0],1),
            nn.ReLU()
        )

        self.pre_interpolate = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channels[i-1],out_channels[i-1],3,1,1,1,out_channels[i-1]),
            nn.Conv2d(out_channels[i-1],out_channels[i],1),
            nn.BatchNorm2d(out_channels[i]),
            nn.ReLU()
        ) for i in range(1, len(out_channels))])
        self.pre_fuse = nn.ModuleList([nn.Sequential(
            nn.Conv2d(i, o, 1),
            nn.BatchNorm2d(o),
            nn.ReLU()
        ) for (i, o) in zip(in_channels,out_channels)])
        self.fuse = nn.ModuleList([nn.Conv2d(2*o, o, 1) for o in out_channels[1:]])
        self.post_fuse = nn.ModuleList([nn.Sequential(*[ConvBlock(o) for _ in range(2)]) for o in out_channels[1:]])

        self.kp_encoder = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channel, per_emb_num * num_joints, 1),
            # nn.BatchNorm2d(per_emb_num * num_joints),
            nn.ReLU()
        ) if supervise else nn.Identity() for out_channel, per_emb_num, supervise in zip(out_channels, per_emb_nums, supervises) ])
        num = len(out_channels)
        self.converge = nn.ModuleList([ConvergeHead(per_emb_nums[i], 2**upsample_log[i], 11-2*(num-i), 5-(num-i), num_joints) if supervises[i] else nn.Identity() for i in range(num)])

    
    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)       # list of tensor
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        """
        下面一串操作就是生成多尺度的金标签的过程
        """
        with torch.no_grad():
            gt_heatmaps_dict = {gt_heatmaps.size(2):gt_heatmaps}
            for output in pred_fields:
                if output.size(2) not in gt_heatmaps_dict:
                    gt_heatmaps_dict[output.size(2)] = F.adaptive_avg_pool2d(gt_heatmaps,output.shape[2:])

        losses = dict()
        for i, output in enumerate(pred_fields):
            if i == 0:
                losses['hp{}_2d_loss'.format(i)] = 0.0001 * self.loss_module(output, gt_heatmaps_dict[output.size(2)], keypoint_weights)       # 作者的预设值。给了很低的权重
            else:
                losses['hp{}_2d_loss'.format(i)] = self.loss_module(output, gt_heatmaps_dict[output.size(2)], keypoint_weights)
        return losses     
    
    
    
    
    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds
        
    
    
    
    def old_loss(self, outputs, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            outputs (Sequence[torch.Tensor[NxKxHxW]]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()
        """
        下面一串操作就是生成多尺度的金标签的过程
        """
        with torch.no_grad():
            targets = {target.size(2):target}
            for output in outputs:
                if output.size(2) not in targets:
                    targets[output.size(2)] = F.adaptive_avg_pool2d(target,output.shape[2:])

        for i, output in enumerate(outputs):
            if i == 0:
                losses['hp{}_2d_loss'.format(i)] = 0.0001 * self.loss_module(output, targets[output.size(2)], target_weight)       # 作者的预设值。给了很低的权重
            else:
                losses['hp{}_2d_loss'.format(i)] = self.loss_module(output, targets[output.size(2)], target_weight)

        return losses

    def get_accuracy(self, outputs, target, target_weight):

        accuracy = dict()

        return accuracy

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        xs = feats

        """original Forward function."""
        xs = xs[::-1]       # 特征图分辨率由小到大排列
        heatmaps = []
        
        for i in range(len(xs)):
            if i == 0:
                hp = self.lr_head(xs[0])            # lr_head只用了这一次，它的作用是直接将最小尺寸的特征图的通道数直接通过1x1卷积变成类别数
                heatmaps.append(hp)     # 保存在heatmaps这个list中 后面要用上深监督
                feat = self.pre_fuse[i](xs[i]) + self.lr_fuse(hp)       # lr_fuse又把通道数上升到了out_channels[0] 然后和一个pre_fuse结果相加
            else:
                feat = self.pre_interpolate[i-1](feat)
                feat = F.interpolate(feat, xs[i].shape[2:])
                feat = torch.cat([feat, self.pre_fuse[i](xs[i])],1)
                feat = self.fuse[i-1](feat)         # 这里就是用 1x1卷积把通道数降到一半
                feat = self.post_fuse[i-1](feat)
            if self.supervises[i] and self.training:
                kp_emb = self.kp_encoder[i](feat)
                hp = self.converge[i](kp_emb)
                heatmaps.append(hp)
        if self.training:
            return heatmaps
        else:
            kp_emb = self.kp_encoder[-1](feat)
            hp = self.converge[-1](kp_emb)
            return hp


                                                                                                                                                                                                                                     

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if self.detector_type == "TopDownHigher":
            if flip_pairs is not None:
                output_flipped = output.detach()
                output_flipped_back = output_flipped.clone()

                # Swap left-right parts
                for left, right in flip_pairs:
                    output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
                    output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
                # Flip horizontally
                output_heatmap = output_flipped_back.flip(3)
                if self.test_cfg.get('shift_heatmap', False):
                    output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, : -1]
            else:
                output_heatmap = output.detach()
        else:
            if flip_pairs is not None:
                output_heatmap = flip_back(
                    output.detach().cpu().numpy(),
                    flip_pairs,
                    target_type=self.target_type)
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if self.test_cfg.get('shift_heatmap', False):
                    output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
            else:
                output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)