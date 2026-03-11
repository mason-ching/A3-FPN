# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from DCNv4 import DCNv4 as dcn_v4
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.config import ConfigDict
# from timm.layers import trunc_normal_
from typing import List, Optional, Union, Tuple, Dict
from torch import Tensor
from ..utils import resize
from mmcv.cnn import build_norm_layer
from ..layers import get_norm, get_activation


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation, normalization).
    """
    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act: Union[bool, nn.Module] = True, bias=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 ):
        """Initialize Conv layer with given arguments including activation and normalization."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        _, self.norm = build_norm_layer(norm_cfg, c2) if isinstance(norm_cfg, (ConfigDict, Dict, dict)) else (
            None, nn.Identity())
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class Identity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Inputs: :math:`List[*]`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the first input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> inputs = [torch.randn(128, 20)]
        >>> output = m(inputs)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, inputs: Union[List[Tensor], Tuple[Tensor]]) -> Tensor:
        return inputs[0]


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W).contiguous()
        return x * self.weight + self.bias


class RepVggBlock(nn.Module):
    def __init__(
            self, ch_in, ch_out, act: Union[str, nn.Module, bool] = 'GELU',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = Conv(ch_in, ch_out, 3, 1, 1, act=False, norm_cfg=norm_cfg)
        self.conv2 = Conv(ch_in, ch_out, 1, 1, 0, act=False, norm_cfg=norm_cfg)
        self.act = nn.Identity() if (act is None or act is False) else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

        # def train(self, mode: bool = True):
        #     r"""Sets the module in training mode.
        #
        #     This has any effect only on certain modules. See documentations of
        #     particular modules for details of their behaviors in training/evaluation
        #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        #     etc.
        #
        #     Args:
        #         mode (bool): whether to set training mode (``True``) or evaluation
        #                      mode (``False``). Default: ``True``.
        #
        #     Returns:
        #         Module: self
        #     """
        #     super().train(mode)
        #     self.convert_to_deploy()

    def convert_to_deploy(self):
        # if self.training:
        #     if hasattr(self, 'conv'):
        #         self.__delattr__('conv')
        # else:
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Conv):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_blocks=1,
            expansion=1.0,
            bias=False,
            act: Union[nn.Module, str] = nn.GELU(),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm_cfg=norm_cfg)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm_cfg=norm_cfg)
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, act=act, norm_cfg=norm_cfg) for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv(hidden_channels, out_channels, 1, 1, bias=True, act=False, norm_cfg=False)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class Fusion(nn.Module):
    """
    Multi-scale Context-aware Attention for Feature Fusion
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_fusion=2,
            compress_c=16,
            act=nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, dict]] = dict(type='SyncBN', requires_grad=True),
            num_blocks: int = 1,
            expansion: float = 1.,
            # dcn arguments
            using_resampling: Optional[bool] = False,
            dcn_group: int = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ):
        super().__init__()

        self.using_resampling = using_resampling

        if num_fusion == 2:
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm_cfg=norm_cfg,
            )

            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm_cfg=norm_cfg
                )
                self.resampler1 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
        elif num_fusion == 3:
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_3 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm_cfg=norm_cfg,
            )
            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm_cfg=norm_cfg
                )
                self.resampler1 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
                self.resampler2 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
        else:
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_3 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_level_4 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm_cfg=norm_cfg,
            )

            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm_cfg=norm_cfg
                )
                self.resampler1 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
                self.resampler2 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
                self.resampler3 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )

        self.conv = Conv(in_channels, out_channels, 1, 1, act=False, norm_cfg=False, bias=True) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):

        if len(x) == 2:
            sampled_feat1, unsampled_feat = x

            # feature_map_visualization(sampled_feat1, module_type='a3fpn_last_fusion',
            #                           save_dir="feats_visualization_no_resampling")
            # draw_heatmap(sampled_feat1, module_type='a3fpn_last_fusion',
            #              save_dir="heat_map_feats_visualization_no_resampling",
            #              )

            if self.using_resampling:
                context_info1 = self.context_conv(
                    torch.cat(x, 1)
                )
                sampled_feat1 = self.resampler1(context_info1, sampled_feat1)

            # feature_map_visualization(sampled_feat1, module_type='a3fpn_last_fusion',
            #                           save_dir="feats_visualization_with_resampling")
            # draw_heatmap(sampled_feat1, module_type='a3fpn_last_fusion',
            #              save_dir="heat_map_feats_visualization_with_resampling",
            #              )

            level_1_weight_v = self.weight_level_1(sampled_feat1)
            level_2_weight_v = self.weight_level_2(unsampled_feat)

            levels_weight = self.weight_levels(
                torch.cat((level_1_weight_v, level_2_weight_v), 1)
            )
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = (sampled_feat1 * levels_weight[:, 0:1, :, :] + unsampled_feat *
                                 levels_weight[:, 1:2, :, :])

            # feature_map_visualization(fused_out_reduced, module_type='a3fpn_last_fusion',
            #                           save_dir="feats_visualization_with_resampling")
            # draw_heatmap(fused_out_reduced, module_type='a3fpn_last_fusion',
            #              save_dir="heat_map_feats_visualization_with_resampling",
            #              )

        elif len(x) == 3:
            sampled_feat1, sampled_feat2, unsampled_feat = x
            if self.using_resampling:
                context_info1, context_info2 = torch.split(
                    self.context_conv(torch.cat(x, 1)), sampled_feat1.size(1), dim=1,
                )
                sampled_feat1 = self.resampler1(context_info1, sampled_feat1)
                sampled_feat2 = self.resampler2(context_info2, sampled_feat2)

            level_1_weight_v = self.weight_level_1(sampled_feat1)
            level_2_weight_v = self.weight_level_2(sampled_feat2)
            level_3_weight_v = self.weight_level_3(unsampled_feat)

            levels_weight = self.weight_levels(
                torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
            )
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = (
                    sampled_feat1 * levels_weight[:, 0:1, :, :] + sampled_feat2 * levels_weight[:, 1:2, :, :]
                    + unsampled_feat * levels_weight[:, 2:, :, :])
        else:
            sampled_feat1, sampled_feat2, sampled_feat3, unsampled_feat = x
            if self.using_resampling:
                context_info1, context_info2, context_info3 = torch.split(
                    self.context_conv(torch.cat(x, 1)), sampled_feat1.size(1), dim=1,
                )

                # feature_map_visualization(sampled_feat3, module_type='a3fpn_last_fusion',
                #                           save_dir="feats_visualization_no_resampling")
                # draw_heatmap(sampled_feat3, module_type='a3fpn_last_fusion',
                #              save_dir="heat_map_feats_visualization_no_resampling",
                #              )

                sampled_feat1 = self.resampler1(context_info1, sampled_feat1)
                sampled_feat2 = self.resampler2(context_info2, sampled_feat2)
                sampled_feat3 = self.resampler3(context_info3, sampled_feat3)

            # feature_map_visualization(sampled_feat3, module_type='a3fpn_last_fusion',
            #                           save_dir="feats_visualization_with_resampling")
            # draw_heatmap(sampled_feat3, module_type='a3fpn_last_fusion',
            #              save_dir="heat_map_feats_visualization_with_resampling",
            #              )

            level_1_weight_v = self.weight_level_1(sampled_feat1)
            level_2_weight_v = self.weight_level_2(sampled_feat2)
            level_3_weight_v = self.weight_level_3(sampled_feat3)
            level_4_weight_v = self.weight_level_4(unsampled_feat)

            levels_weight = self.weight_levels(
                torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)
            )
            # levels_weight = F.softmax(levels_weight, dim=1)
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = sampled_feat1 * levels_weight[:, 0:1, :, :] + \
                                sampled_feat2 * levels_weight[:, 1:2, :, :] + \
                                sampled_feat3 * levels_weight[:, 2:3, :, :] + \
                                unsampled_feat * levels_weight[:, 3:, :, :]

        return self.conv(fused_out_reduced)


class Reassemble(nn.Module):
    """
    Intra-scale Content-Aware Attention for Feature Reassemble
    """

    def __init__(
            self,
            out_channels: int,
            group_num: int = 16,
            gate_threshold: float = 0.5,
            torch_gn: bool = True,
            act=nn.GELU(),
            norm: Optional[Union[bool, nn.Module, str]] = 'LN',
    ):
        super().__init__()

        if group_num is None:
            group_num = out_channels

        self.gn = nn.GroupNorm(num_channels=out_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=out_channels, group_num=group_num)

        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()
        self.act = get_activation(act=act)
        self.norm = nn.Identity() if norm is False else get_norm(norm, out_channels)

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1).contiguous()
        reweigts = self.sigmoid(gn_x * w_gamma)
        # Threshold
        w1 = torch.where(reweigts > self.gate_threshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_threshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.channel_reassemblle(x_1, x_2)
        return self.act(self.norm(y))

    @staticmethod
    def channel_reassemblle(x_1, x_2):
        return x_1 + x_2.flip(1).contiguous()


class SampleBlock(nn.Module):
    """
    sampling module
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            sample: str = 'upsample',
            scale_factor: int = 2,
            upsample_method: str = 'bilinear',
            align_corners: Optional[bool] = False,
            group: int = 1,
            act: Optional[Union[bool, nn.Module]] = nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, str]] = dict(type='SyncBN', requires_grad=True),
    ) -> None:
        """
        Args:
            in_channels:
            out_channels:
            sample:
            scale_factor:
            upsample_method:
            align_corners:
            group:
            act:
            norm_cfg:
        Returns: None
        """
        super().__init__()

        if sample == 'upsample':
            self.sample_layer = nn.Sequential(
                *[Conv(in_channels, out_channels, 1, act=act,
                       norm_cfg=norm_cfg) if in_channels != out_channels else nn.Identity()],
                nn.Upsample(scale_factor=scale_factor, mode=upsample_method, align_corners=align_corners),
            )
        elif sample == 'downsample':
            self.sample_layer = Conv(in_channels, out_channels, scale_factor, scale_factor, 0,
                                     g=in_channels if in_channels == out_channels else group,
                                     act=act, norm_cfg=norm_cfg)
        else:
            raise ValueError('your sample option must be \'upsample\' or \'downsample\'')

    def forward(self, feat_s: Tensor) -> Tensor:
        """
        Args:
            feat_s (Tensor): feature maps that need to be sampled
        return: Tensor
        """
        sampled_feat = self.sample_layer(feat_s)
        # if sampled_feat.shape[2:] != feat_u.shape[2:]:
        #     sampled_feat = resize(sampled_feat, feat_u.shape[2:])
        return sampled_feat


class Resampler(nn.Module):
    """
    context-aware resampling by dynamic and sparse convolutions
    """

    def __init__(
            self,
            channels: int,
            act: Optional[Union[bool, nn.Module]] = nn.GELU(),
            dcn_norm: Optional[Union[bool, nn.Module, str]] = "LN",
            dcn_group: int = 4,
            offset_scale: int = 0.5,
            dw_kernel_size: int = 3,
            dcn_output_bias: bool = False,
            center_feature_scale: bool = False,
            remove_center: bool = False,
            without_pointwise: bool = False,
    ) -> None:
        """
        Args:
             channels:
             act:
             dcn_norm:
             dcn_group:
             offset_scale:
             dw_kernel_size:
             dcn_output_bias:
             center_feature_scale:
             remove_center:
             without_pointwise:
        return None
        """
        super(Resampler, self).__init__()

        self.resampling = dcn_v4(
            channels, 3, stride=1, pad=1, dilation=1, group=dcn_group, offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            output_bias=dcn_output_bias,
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,
            without_pointwise=without_pointwise,
            extra_offset_mask=True
        )
        self.norm = get_norm(dcn_norm, channels) if dcn_norm else nn.Identity()
        self.act = nn.GELU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, context_info: Tensor, sampled_feat: Tensor, visualize: bool = False) -> Tensor:
        """
        Args:
            context_info (Tensor): context information
            sampled_feat (Tensor): feature maps that have been sampled
        return: Tensor
        """
        # feature_map_visualization(sampled_feat, module_type='a3fpnv4_last_level', save_dir="feats_visualization_sampling")

        adjusted_features = self.act(
            self.norm(
                self.resampling([sampled_feat, context_info], visualize)  # [sampled_feat, offset]
            )
        )
        # feature_map_visualization(adjusted_features, module_type='a3fpnv4_last_level', save_dir="feats_visualization_resampling")
        return adjusted_features


class A3FPN_2(nn.Module):
    def __init__(
            self,
            level=0,
            channel=None,
            act=nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, str]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: int = 16,
            group_num: int = 16,
            num_blocks: int = 1,
            expansion: float = 1.,
            using_resampling: Optional[bool] = False,
            # dcn arguments
            dcn_group: int = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ):
        super(A3FPN_2, self).__init__()

        if channel is None:
            channel = [64, 128]
        self.level = level
        if self.level == 0:
            self.upsample = SampleBlock(
                channel[1], channel[0], sample='upsample', scale_factor=2, act=act,
                norm_cfg=norm_cfg,
            )
        else:
            self.downsample = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2, act=act,
                norm_cfg=norm_cfg,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=2, compress_c=compress_channel,
            act=act, norm_cfg=norm_cfg, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
            dcn_group=dcn_group, dcn_config=dcn_config,
        )
        self.ICA = Reassemble(
            out_channels=channel[level], group_num=group_num, act=act, norm='LN',
        )

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
            out = self.MCA((input2, input1))

        elif self.level == 1:
            input1 = self.downsample(input1)
            out = self.MCA((input1, input2))

        out = self.ICA(out)
        return out


class A3FPN_3(nn.Module):
    def __init__(
            self,
            level=0,
            channel=None,
            act=nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, str]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: int = 16,
            group_num: int = 16,
            num_blocks: int = 1,
            expansion: float = 1.,
            using_resampling: Optional[bool] = False,
            # dcn arguments
            dcn_group: Union[int, List[int], Tuple[int]] = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ):
        super(A3FPN_3, self).__init__()

        if channel is None:
            channel = [64, 128, 256]

        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(
                channel[2], channel[0], sample='upsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )

        elif self.level == 2:
            self.downsample2x = SampleBlock(
                channel[1], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample4x = SampleBlock(
                channel[0], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=3, compress_c=compress_channel,
            act=act, norm_cfg=norm_cfg, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
            dcn_group=dcn_group, dcn_config=dcn_config,
        )
        self.ICA = Reassemble(
            out_channels=channel[level], group_num=group_num, act=act, norm='LN',
        )

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            out = self.MCA((input2, input3, input1))

        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
            # upsample, downsample, unsample
            out = self.MCA((input3, input1, input2))

        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
            out = self.MCA((input2, input1, input3))

        out = self.ICA(out)
        return out


class A3FPN_4(nn.Module):
    def __init__(
            self,
            level=0,
            channel=None,
            act=nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, str]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: int = 16,
            group_num: int = 16,
            num_blocks: int = 1,
            expansion: float = 1.,
            using_resampling: Optional[bool] = False,
            # dcn arguments
            dcn_group: Union[int, List[int], Tuple[int]] = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ):
        super(A3FPN_4, self).__init__()
        if channel is None:
            channel = [256, 512, 1024, 2048]
        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(
                channel[2], channel[0], sample='upsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.upsample8x = SampleBlock(
                channel[3], channel[0], sample='upsample', scale_factor=8,
                act=act,
                norm_cfg=norm_cfg,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.upsample4x1 = SampleBlock(
                channel[3], channel[1], sample='upsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )

        elif self.level == 2:
            self.upsample2x2 = SampleBlock(
                channel[3], channel[2], sample='upsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample2x2 = SampleBlock(
                channel[1], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample4x2 = SampleBlock(
                channel[0], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )

        elif self.level == 3:
            self.downsample2x3 = SampleBlock(
                channel[2], channel[3], sample='downsample', scale_factor=2,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample4x3 = SampleBlock(
                channel[1], channel[3], sample='downsample', scale_factor=4,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.downsample8x3 = SampleBlock(
                channel[0], channel[3], sample='downsample', scale_factor=8,
                act=act,
                norm_cfg=norm_cfg,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=4, compress_c=compress_channel,
            act=act, norm_cfg=norm_cfg, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
            dcn_group=dcn_group, dcn_config=dcn_config,
        )
        self.ICA = Reassemble(
            out_channels=channel[level], group_num=group_num, act=act, norm='LN',
        )

    def forward(self, x):
        input1, input2, input3, input4 = x

        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input4 = self.upsample8x(input4)
            # upsample, downsample, unsample
            out = self.MCA((input2, input3, input4, input1))

        elif self.level == 1:
            input1 = self.downsample2x1(input1)
            input3 = self.upsample2x1(input3)
            input4 = self.upsample4x1(input4)
            out = self.MCA((input3, input4, input1, input2))

        elif self.level == 2:
            input1 = self.downsample4x2(input1)
            input2 = self.downsample2x2(input2)
            input4 = self.upsample2x2(input4)
            # upsample, downsample, unsample
            out = self.MCA((input4, input2, input1, input3))

        elif self.level == 3:
            input1 = self.downsample8x3(input1)
            input2 = self.downsample4x3(input2)
            input3 = self.downsample2x3(input3)
            # upsample, downsample, unsample
            out = self.MCA((input3, input2, input1, input4))

        out = self.ICA(out)

        return out


class Body(nn.Module):
    def __init__(
            self,
            channels=None,
            act: nn.Module = nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, str]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            group_num: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            num_repblocks: int = 1,
            expansion: float = 1.,
            using_resampling: Union[List[bool], Tuple[bool]] = False,
            # dcn arguments
            dcn_groups: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ):
        super(Body, self).__init__()

        if channels is None:
            channels = [64, 128, 256, 512]

        # top-down
        # self.a3fpn_2_level0 = A3FPN_2(
        #     level=0, channel=channels[-2:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-2],
        #     group_num=group_num[-2],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[0],
        #     dcn_group=dcn_groups[-2],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_2_level1 = A3FPN_2(
        #     level=1, channel=channels[-2:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-1],
        #     group_num=group_num[-1],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[0],
        #     dcn_group=dcn_groups[-1],
        #     dcn_config=dcn_config,
        # )
        #
        # self.a3fpn_3_level0 = A3FPN_3(
        #     level=0, channel=channels[-3:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-3],
        #     group_num=group_num[-3],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[-3],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_3_level1 = A3FPN_3(
        #     level=1, channel=channels[-3:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-2],
        #     group_num=group_num[-2],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[-2],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_3_level2 = A3FPN_3(
        #     level=2, channel=channels[-3:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-1],
        #     group_num=group_num[-1],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[-1],
        #     dcn_config=dcn_config,
        # )
        #
        # self.a3fpn_4_level0 = A3FPN_4(
        #     level=0, channel=channels[-4:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-4],
        #     group_num=group_num[-4],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[-4],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level1 = A3FPN_4(
        #     level=1, channel=channels[-4:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-3],
        #     group_num=group_num[-3],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[-3],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level2 = A3FPN_4(
        #     level=2, channel=channels[-4:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-2],
        #     group_num=group_num[-2],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[-2],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level3 = A3FPN_4(
        #     level=3, channel=channels[-4:],
        #     act=act,
        #     norm_cfg=norm_cfg,
        #     compress_channel=compress_channel[-1],
        #     group_num=group_num[-1],
        #     num_blocks=num_repblocks,
        #     expansion=expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[-1],
        #     dcn_config=dcn_config,
        # )

        # bottom-up
        self.a3fpn_2_level0 = A3FPN_2(
            level=0, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[0],
            group_num=group_num[0],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[0],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.a3fpn_2_level1 = A3FPN_2(
            level=1, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[1],
            group_num=group_num[1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[0],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )

        self.a3fpn_3_level0 = A3FPN_3(
            level=0, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[0],
            group_num=group_num[0],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level1 = A3FPN_3(
            level=1, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[1],
            group_num=group_num[1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level2 = A3FPN_3(
            level=2, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[2],
            group_num=group_num[2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[2],
            dcn_config=dcn_config,
        )

        self.a3fpn_4_level0 = A3FPN_4(
            level=0, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[0],
            group_num=group_num[0],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level1 = A3FPN_4(
            level=1, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[1],
            group_num=group_num[1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level2 = A3FPN_4(
            level=2, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[2],
            group_num=group_num[2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[2],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level3 = A3FPN_4(
            level=3, channel=channels,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[3],
            group_num=group_num[3],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[3],
            dcn_config=dcn_config,
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        # bottom-up
        x1 = self.a3fpn_2_level1((x0, x1))
        output0 = self.a3fpn_2_level0((x0, x1))

        x0 = self.a3fpn_3_level0((output0, x1, x2))
        output1 = self.a3fpn_3_level1((output0, x1, x2))
        x2 = self.a3fpn_3_level2((output0, x1, x2))

        output0 = self.a3fpn_4_level0((x0, output1, x2, x3))
        x1 = self.a3fpn_4_level1((x0, output1, x2, x3))
        output2 = self.a3fpn_4_level2((x0, output1, x2, x3))
        x3 = self.a3fpn_4_level3((x0, output1, x2, x3))

        # top-down
        # output2 = self.a3fpn_2_level0((x2, x3))
        # x3 = self.a3fpn_2_level1((x2, x3))
        #
        # output1 = self.a3fpn_3_level0((x1, output2, x3))
        # x2 = self.a3fpn_3_level1((x1, output2, x3))
        # x3 = self.a3fpn_3_level2((x1, output2, x3))
        #
        # output0 = self.a3fpn_4_level0((x0, output1, x2, x3))
        # x1 = self.a3fpn_4_level1((x0, output1, x2, x3))
        # output2 = self.a3fpn_4_level2((x0, output1, x2, x3))
        # x3 = self.a3fpn_4_level3((x0, output1, x2, x3))

        outputs = [output0, x1, output2, x3]

        return outputs


@MODELS.register_module()
class A3FPN(BaseModule):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            squeeze: Union[List[int], int] = 1,
            act=nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, dict]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            group_num: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            num_repblocks: int = 1,
            expansion: float = 1.,
            using_resampling: Union[List[bool], Tuple[bool]] = False,
            # dcn arguments
            dcn_groups: Union[int, List[int], Tuple[int]] = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
            init_cfg=None,
    ):
        super(A3FPN, self).__init__(init_cfg)

        if isinstance(squeeze, int):
            squeeze = [squeeze] * len(in_channels)
        if isinstance(dcn_groups, int):
            dcn_groups = [dcn_groups] * len(in_channels)

        in_channels_reduced = [channel // squeeze[i] for i, channel in enumerate(in_channels)]
        # self.fp16_enabled = False

        self.conv0 = Conv(in_channels[0], in_channels_reduced[0], 1, act=act, norm_cfg=norm_cfg, bias=False) \
            if squeeze[0] != 1 else nn.Identity()
        self.conv1 = Conv(in_channels[1], in_channels_reduced[1], 1, act=act, norm_cfg=norm_cfg, bias=False) \
            if squeeze[1] != 1 else nn.Identity()
        self.conv2 = Conv(in_channels[2], in_channels_reduced[2], 1, act=act, norm_cfg=norm_cfg, bias=False) \
            if squeeze[2] != 1 else nn.Identity()
        self.conv3 = Conv(in_channels[3], in_channels_reduced[3], 1, act=act, norm_cfg=norm_cfg, bias=False) \
            if squeeze[3] != 1 else nn.Identity()

        self.a3fpn_body = Body(
            channels=in_channels_reduced,
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel,
            group_num=group_num,
            num_repblocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )

        self.conv00 = Conv(in_channels_reduced[0], out_channels, 3, p=1, act=act, norm_cfg=norm_cfg)
        self.conv11 = Conv(in_channels_reduced[1], out_channels, 3, p=1, act=act, norm_cfg=norm_cfg)
        self.conv22 = Conv(in_channels_reduced[2], out_channels, 3, p=1, act=act, norm_cfg=norm_cfg)
        self.conv33 = Conv(in_channels_reduced[3], out_channels, 3, p=1, act=act, norm_cfg=norm_cfg)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, dcn_v4):
            m._reset_parameters()

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.a3fpn_body((x0, x1, x2, x3))

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return [out0, out1, out2, out3]

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        # return self


import os
import torch
import matplotlib.pyplot as plt
import math
import logging as LOGGER
import numpy as np
from pathlib import Path
from matplotlib.image import imsave
from PIL import Image, ImageEnhance
import cv2
from detectron2.data.detection_utils import read_image


def featuremap_2_heatmap(feature_map):
    # assert isinstance(feature_map, torch.Tensor)

    # 1*256*200*256 # feat的维度要求，四维
    # feature_map = feature_map.detach()
    #
    # # 1*256*200*256->1*200*256
    # heatmap = feature_map[:, 0, :, :] * 0
    # for c in range(feature_map.shape[1]):
    #     heatmap += feature_map[:, c, :, :]
    # heatmap = heatmap.cpu().numpy()
    # heatmap = np.mean(heatmap, axis=1)

    heatmap = np.maximum(feature_map, 1e-8)
    # max_value = np.max(heatmap)
    # if max_value != 0:
    #     heatmap /= max_value
    # else:
    #     print("max_value is 0")
    #     heatmap /= 1e-8
    heatmap /= np.max(heatmap)

    return heatmap


def draw_heatmap(
        featuremap: Union[Tensor, List[Tensor], Tuple[Tensor]],
        module_type: str,
        save_dir: str = "feats_visualization",
        img_path: str = '/media/mengen/T7/public-data/CityScapes/leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png',
        only_save_merged: bool = False,
        average_merge: bool = True,
):
    # args = get_parser().parse_args()
    # cfg = setup_cfg(args)
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # from predictor import VisualizationDemo
    # demo = VisualizationDemo(cfg)
    # for imgs in tqdm.tqdm(os.listdir(img_path)):
    img = read_image(os.path.join(img_path, img_path), format="BGR")
    #     start_time = time.time()
    #     predictions = demo.run_on_image(img)  # 后面需对网络输出做一定修改，
    #     # 会得到一个字典P3-P7的输出
    #     logger.info(
    #         "{}: detected in {:.2f}s".format(
    #             imgs, time.time() - start_time))
    #     i = 0
    #     for featuremap in list(predictions.values()):
    if isinstance(featuremap, Tensor):
        featuremap = [featuremap]
    for j, feat in enumerate(featuremap):
        channels = feat.shape[1]
        feature_map = torch.squeeze(feat)
        feature_map = feature_map.detach().cpu().numpy()

        feature_map_sum = feature_map[0, :, :]
        # feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
        for i in range(0, channels):
            # out_path = save_dir / f"output{j + 1}/{module_type.split('.')[-1]}_features_channel{i + 1}.png"  # out_path
            out_path = save_dir + f"/{module_type.split('.')[-1]}_features_channel{i}.png"  # out_path
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            feature_map_split = feature_map[i, :, :]
            # feature_map_split = np.expand_dims(feature_map_split, axis=2)
            if i > 0:
                feature_map_sum += feature_map_split

            if only_save_merged:
                continue
            heatmap = featuremap_2_heatmap(feature_map_split)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式512*640*3
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
            superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，得到合适的热力图
            cv2.imwrite(out_path,
                        superimposed_img)  # 将图像保存

        out_path = save_dir + f"/{module_type.split('.')[-1]}_features_merged-channel.png"

        # os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if average_merge:
            feature_map_sum /= channels

        heatmap = featuremap_2_heatmap(feature_map_sum)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式512*640*3
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，得到合适的热力图
        cv2.imwrite(out_path,
                    superimposed_img)  # 将图像保存


def feature_map_visualization(
        x: Union[Tensor, List[Tensor], Tuple[Tensor]],
        module_type: str,
        save_dir: str = "feats_visualization",
        using_interpolation: bool = False,
        only_save_merged: bool = False,
        average_merge: bool = True,
        scales: Union[list, tuple] = (4, 8, 16, 32, 64),
) -> None:
    if isinstance(x, Tensor):
        x = [x]
    save_dir = Path(save_dir)
    assert x[0].shape[0] == 1, \
        'feature map visualization now only supports one image each time, batch size should be 1.'

    for j, feat in enumerate(x):
        BI = BilinearInterpolation(scales[j], scales[j]) if using_interpolation else lambda feats: feats
        channels = feat.shape[1]
        feature_map = torch.squeeze(feat)
        feature_map = feature_map.detach().cpu().numpy()

        feature_map_sum = feature_map[0, :, :]
        # feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
        for i in range(0, channels):

            # out_path = save_dir / f"output{j + 1}/{module_type.split('.')[-1]}_features_channel{i + 1}.png"  # out_path
            out_path = save_dir / f"{module_type.split('.')[-1]}_features_channel{i + 1}.png"  # out_path

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            feature_map_split = feature_map[i, :, :]
            # feature_map_split = np.expand_dims(feature_map_split, axis=2)
            if i > 0:
                feature_map_sum += feature_map_split
                feature_map_split = BI(feature_map_split)

            if only_save_merged:
                continue

            # feature_map_split = feature_map_split.astype(np.uint8)
            # imsave(out_path, feature_map_split, vmin=0, vmax=255)

            # cv2.imwrite(out_path, convert_grayscale_to_color(feature_map_split, colormap))

            # img = Image.fromarray(feature_map_split)
            # img = img.convert('RGB')
            # enhancer = ImageEnhance.Brightness(img)
            # img = enhancer.enhance(2)
            # img.save(out_path)

            plt.imshow(
                feature_map_split,
                cmap='gray',
            )
            LOGGER.info(f"Saving {out_path}... ({i + 1}/{channels})")

            plt.axis('off')
            plt.savefig(out_path, dpi=100)
            plt.close()

            # np.save(str(out_path.with_suffix(".npy")), feature_map_split)  # npy save

        # out_path = save_dir / f"output{j + 1}/{module_type.split('.')[-1]}_features_merged-channel.png"
        out_path = save_dir / f"{module_type.split('.')[-1]}_features_merged-channel.png"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        feature_map_sum = BI(feature_map_sum)

        if average_merge:
            feature_map_sum /= channels

        # feature_map_sum = feature_map_sum.astype(np.uint8)
        # imsave(out_path, feature_map_sum, vmin=0, vmax=255)

        # cv2.imwrite(out_path, convert_grayscale_to_color(feature_map_sum, colormap))

        # img = Image.fromarray(feature_map_sum)
        # img = img.convert('RGB')
        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(2)
        # img.save(out_path)

        plt.imshow(
            feature_map_sum,
            cmap='gray',
        )
        plt.axis('off')
        LOGGER.info(f"Saving {out_path}... (channel-merged feature map)")
        plt.savefig(out_path, dpi=100)
        plt.close()

        # np.save(str(out_path.with_suffix(".npy")), feature_map_sum)  # npy save


class BilinearInterpolation(object):
    """
    Bi-linear interpolation method
    """

    def __init__(self, w_rate: float, h_rate: float, *, align: str = 'center') -> None:
        if align not in ['center', 'left']:
            LOGGER.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self, w_rate: float, h_rate: float) -> None:
        self.w_rate = w_rate  # w 的缩放率
        self.h_rate = h_rate  # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i: float, source_h: float, goal_h: float) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h / goal_h))
        else:
            # center 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h / goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i

    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j: float, source_w: float, goal_w: float) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w / goal_w))
        else:
            # center 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w / goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def __call__(self, img: np.ndarray, *args, **kwargs) -> np.ndarray:
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):  # h
            src_i = self.get_src_h(i, source_h, goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j, source_w, goal_w)
                i2 = math.ceil(src_i)
                i1 = int(src_i)
                j2 = math.ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1] * x2_x * y2_y + img[i1, j2] * \
                                x_x1 * y2_y + img[i2, j1] * x2_x * y_y1 + img[i2, j2] * x_x1 * y_y1
        return new_img
