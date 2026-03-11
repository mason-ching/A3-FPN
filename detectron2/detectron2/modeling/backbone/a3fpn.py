import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
import torch.nn as nn

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from typing import List, Optional, Union, Tuple
from torch import Tensor
from DCNv4 import DCNv4 as dcn_v4
from detectron2.layers import (ShapeSpec, get_norm, get_activation)

__all__ = ["build_resnet_a3fpn_backbone", "A3FPN_backbone", 'A3FPN']


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
                 norm: Union[bool, nn.Module, str] = 'SyncBN',
                 ):
        """Initialize Conv layer with given arguments including activation and normalization."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.norm = get_norm(norm, c2) if isinstance(norm, str) else nn.Identity()
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
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class RepVggBlock(nn.Module):
    def __init__(
            self, ch_in, ch_out, act: Union[str, nn.Module, bool] = 'GELU',
            norm='SyncBN',
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = Conv(ch_in, ch_out, 3, 1, 1, act=False, norm=norm)
        self.conv2 = Conv(ch_in, ch_out, 1, 1, 0, act=False, norm=norm)
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
            norm='SyncBN',
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm=norm)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm=norm)
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, act=act, norm=norm) for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv(hidden_channels, out_channels, 1, 1, bias=True, act=False, norm=False)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


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
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm=norm,
            )

            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm=norm
                )
                self.resampler1 = Resampler(
                    channels=in_channels,
                    act=act,
                    dcn_group=dcn_group,
                    **dcn_config
                )
        elif num_fusion == 3:
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_3 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm=norm,
            )
            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm=norm
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
            self.weight_level_1 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_2 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_3 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_level_4 = Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm)
            self.weight_levels = CSPRepLayer(
                in_channels=compress_c * num_fusion, out_channels=num_fusion, num_blocks=num_blocks,
                expansion=expansion, act=act, norm=norm,
            )

            if using_resampling:
                self.context_conv = Conv(
                    in_channels * num_fusion, in_channels * (num_fusion - 1),
                    k=1, s=1, p=0, bias=False, act=act, norm=norm
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

        self.conv = Conv(in_channels, out_channels, 1, 1, act=False, norm=False, bias=True) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):

        if len(x) == 2:
            sampled_feat1, unsampled_feat = x
            if self.using_resampling:
                context_info1 = self.context_conv(
                    torch.cat(x, 1)
                )
                sampled_feat1 = self.resampler1(context_info1, sampled_feat1)

            level_1_weight_v = self.weight_level_1(sampled_feat1)
            level_2_weight_v = self.weight_level_2(unsampled_feat)

            levels_weight = self.weight_levels(
                torch.cat((level_1_weight_v, level_2_weight_v), 1)
            )
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = (sampled_feat1 * levels_weight[:, 0:1, :, :] + unsampled_feat *
                                 levels_weight[:, 1:2, :, :])
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
                sampled_feat1 = self.resampler1(context_info1, sampled_feat1)
                sampled_feat2 = self.resampler2(context_info2, sampled_feat2)
                sampled_feat3 = self.resampler3(context_info3, sampled_feat3)

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
        y = self.channel_reassemble(x_1, x_2)
        return self.act(self.norm(y))

    @staticmethod
    def channel_reassemble(x_1, x_2):
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
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
            norm:
        Returns: None
        """
        super().__init__()

        if sample == 'upsample':
            self.sample_layer = nn.Sequential(
                *[Conv(in_channels, out_channels, 1, act=act,
                       norm=norm) if in_channels != out_channels else nn.Identity()],
                nn.Upsample(scale_factor=scale_factor, mode=upsample_method, align_corners=align_corners),
            )
        elif sample == 'downsample':
            self.sample_layer = Conv(in_channels, out_channels, scale_factor, scale_factor, 0,
                                     g=in_channels if in_channels == out_channels else group,
                                     act=act, norm=norm)
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

    def forward(self, context_info: Tensor, sampled_feat: Tensor) -> Tensor:
        """
        Args:
            context_info (Tensor): context information
            sampled_feat (Tensor): feature maps that have been sampled
        return: Tensor
        """

        adjusted_features = self.act(
            self.norm(
                self.resampling([sampled_feat, context_info])  # [sampled_feat, offset]
            )
        )
        return adjusted_features


class A3FPN_2(nn.Module):
    def __init__(
            self,
            level=0,
            channel=None,
            act=nn.GELU(),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
                norm=norm,
            )
        else:
            self.downsample = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2, act=act,
                norm=norm,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=2, compress_c=compress_channel,
            act=act, norm=norm, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
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
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
                norm=norm,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
            )

        elif self.level == 2:
            self.downsample2x = SampleBlock(
                channel[1], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.downsample4x = SampleBlock(
                channel[0], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=3, compress_c=compress_channel,
            act=act, norm=norm, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
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
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
                norm=norm,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.upsample8x = SampleBlock(
                channel[3], channel[0], sample='upsample', scale_factor=8,
                act=act,
                norm=norm,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.upsample4x1 = SampleBlock(
                channel[3], channel[1], sample='upsample', scale_factor=4,
                act=act,
                norm=norm,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
            )

        elif self.level == 2:
            self.upsample2x2 = SampleBlock(
                channel[3], channel[2], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.downsample2x2 = SampleBlock(
                channel[1], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.downsample4x2 = SampleBlock(
                channel[0], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
            )

        elif self.level == 3:
            self.downsample2x3 = SampleBlock(
                channel[2], channel[3], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
            )
            self.downsample4x3 = SampleBlock(
                channel[1], channel[3], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
            )
            self.downsample8x3 = SampleBlock(
                channel[0], channel[3], sample='downsample', scale_factor=8,
                act=act,
                norm=norm,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=4, compress_c=compress_channel,
            act=act, norm=norm, num_blocks=num_blocks, expansion=expansion, using_resampling=using_resampling,
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
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
        self.a3fpn_2_level0 = A3FPN_2(
            level=0, channel=channels[-2:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-2],
            group_num=group_num[-2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[0],
            dcn_group=dcn_groups[-2],
            dcn_config=dcn_config,
        )
        self.a3fpn_2_level1 = A3FPN_2(
            level=1, channel=channels[-2:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-1],
            group_num=group_num[-1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[0],
            dcn_group=dcn_groups[-1],
            dcn_config=dcn_config,
        )

        self.a3fpn_3_level0 = A3FPN_3(
            level=0, channel=channels[-3:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-3],
            group_num=group_num[-3],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[-3],
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level1 = A3FPN_3(
            level=1, channel=channels[-3:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-2],
            group_num=group_num[-2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[-2],
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level2 = A3FPN_3(
            level=2, channel=channels[-3:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-1],
            group_num=group_num[-1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[1],
            dcn_group=dcn_groups[-1],
            dcn_config=dcn_config,
        )

        self.a3fpn_4_level0 = A3FPN_4(
            level=0, channel=channels[-4:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-4],
            group_num=group_num[-4],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[-4],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level1 = A3FPN_4(
            level=1, channel=channels[-4:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-3],
            group_num=group_num[-3],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[-3],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level2 = A3FPN_4(
            level=2, channel=channels[-4:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-2],
            group_num=group_num[-2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[-2],
            dcn_config=dcn_config,
        )
        self.a3fpn_4_level3 = A3FPN_4(
            level=3, channel=channels[-4:],
            act=act,
            norm=norm,
            compress_channel=compress_channel[-1],
            group_num=group_num[-1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling[2],
            dcn_group=dcn_groups[-1],
            dcn_config=dcn_config,
        )

        # bottom-up
        # self.a3fpn_2_level0 = A3FPN_2(
        #     level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[0],
        #     group_num = group_num[0],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[0],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_2_level1 = A3FPN_2(
        #     level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[1],
        #     group_num = group_num[1],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[0],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        #
        # self.a3fpn_3_level0 = A3FPN_3(
        #     level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[0],
        #     group_num = group_num[0],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_3_level1 = A3FPN_3(
        #     level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[1],
        #     group_num = group_num[1],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_3_level2 = A3FPN_3(
        #     level=2, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[2],
        #     group_num = group_num[2],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[1],
        #     dcn_group=dcn_groups[2],
        #     dcn_config=dcn_config,
        # )
        #
        # self.a3fpn_4_level0 = A3FPN_4(
        #     level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[0],
        #     group_num = group_num[0],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level1 = A3FPN_4(
        #     level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[1],
        #     group_num = group_num[1],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level2 = A3FPN_4(
        #     level=2, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[2],
        #     group_num = group_num[2],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[2],
        #     dcn_config=dcn_config,
        # )
        # self.a3fpn_4_level3 = A3FPN_4(
        #     level=3, channel=channels,
        #     act=act,
        #     norm=norm,
        #     compress_channel = compress_channel[3],
        #     group_num = group_num[3],
        #     num_blocks = num_repblocks,
        #     expansion = expansion,
        #     using_resampling=using_resampling[2],
        #     dcn_group=dcn_groups[3],
        #     dcn_config=dcn_config,
        # )

    def forward(self, x):
        x0, x1, x2, x3 = x

        # bottom-up
        # output0 = self.a3fpn_2_level0((x0, x1))
        # x1 = self.a3fpn_2_level1((x0, x1))
        #
        # x0 = self.a3fpn_3_level0((output0, x1, x2))
        # output1 = self.a3fpn_3_level1((output0, x1, x2))
        # x2 = self.a3fpn_3_level2((output0, x1, x2))
        #
        # output0 = self.a3fpn_4_level0((x0, output1, x2, x3))
        # x1 = self.a3fpn_4_level1((x0, output1, x2, x3))
        # output2 = self.a3fpn_4_level2((x0, output1, x2, x3))
        # x3 = self.a3fpn_4_level3((x0, output1, x2, x3))
        #
        # outputs = [output0, x1, output2, x3]

        # top-down
        output2 = self.a3fpn_2_level0((x2, x3))
        x3 = self.a3fpn_2_level1((x2, x3))

        output1 = self.a3fpn_3_level0((x1, output2, x3))
        x2 = self.a3fpn_3_level1((x1, output2, x3))
        x3 = self.a3fpn_3_level2((x1, output2, x3))

        output0 = self.a3fpn_4_level0((x0, output1, x2, x3))
        x1 = self.a3fpn_4_level1((x0, output1, x2, x3))
        output2 = self.a3fpn_4_level2((x0, output1, x2, x3))
        x3 = self.a3fpn_4_level3((x0, output1, x2, x3))

        outputs = [output0, x1, output2, x3]

        return outputs


class A3FPN(nn.Module):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            squeeze: Union[List[int], int] = 1,
            act=nn.GELU(),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
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
    ):
        super(A3FPN, self).__init__()

        if isinstance(squeeze, int):
            squeeze = [squeeze] * len(in_channels)
        if isinstance(dcn_groups, int):
            dcn_groups = [dcn_groups] * len(in_channels)

        in_channels_reduced = [channel // squeeze[i] for i, channel in enumerate(in_channels)]
        # self.fp16_enabled = False

        self.conv0 = Conv(in_channels[0], in_channels_reduced[0], 1, act=act, norm=norm, bias=False) \
            if squeeze[0] != 1 else nn.Identity()
        self.conv1 = Conv(in_channels[1], in_channels_reduced[1], 1, act=act, norm=norm, bias=False) \
            if squeeze[1] != 1 else nn.Identity()
        self.conv2 = Conv(in_channels[2], in_channels_reduced[2], 1, act=act, norm=norm, bias=False) \
            if squeeze[2] != 1 else nn.Identity()
        self.conv3 = Conv(in_channels[3], in_channels_reduced[3], 1, act=act, norm=norm, bias=False) \
            if squeeze[3] != 1 else nn.Identity()

        self.a3fpn_body = Body(
            channels=in_channels_reduced,
            act=act,
            norm=norm,
            compress_channel=compress_channel,
            group_num=group_num,
            num_repblocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )

        self.conv00 = Conv(in_channels_reduced[0], out_channels, 3, p=1, act=act, norm=norm)
        self.conv11 = Conv(in_channels_reduced[1], out_channels, 3, p=1, act=act, norm=norm)
        self.conv22 = Conv(in_channels_reduced[2], out_channels, 3, p=1, act=act, norm=norm)
        self.conv33 = Conv(in_channels_reduced[3], out_channels, 3, p=1, act=act, norm=norm)

        # # init weight
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight, gain=0.02)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        #         torch.nn.init.constant_(m.bias.data, 0.0)

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


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    @staticmethod
    def forward(x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class A3FPN_backbone(Backbone):
    """
    This module implements the paper:`A3-FPN`.
    """

    def __init__(
            self,
            bottom_up,
            in_features,
            out_channels,
            top_block=None,
            squeeze: Union[List[int], Tuple[int], int] = [1, 2, 2, 4],
            act: nn.Module = nn.GELU(),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            compress_channel: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            group_num: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            num_repblocks: int = 1,
            expansion: float = 1.,
            using_resampling: Union[List[bool], Tuple[bool]] = False,
            # dcn arguments
            dcn_groups: Union[List[int], Tuple[int]] = 4,
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
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate A3FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which A3FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                A3FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra A3FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
        """
        super(A3FPN_backbone, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self.a3fpn = A3FPN(
            in_channels=in_channels_per_feature,
            out_channels=out_channels,
            squeeze=squeeze,
            act=act,
            norm=norm,
            compress_channel=compress_channel,
            group_num=group_num,
            num_repblocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )
        stage = int(math.log2(strides[len(in_channels_per_feature) - 1]))
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.

        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

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

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            x (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to A3FPN feature map tensor
                in high to low resolution order. Returned feature names follow the A3FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = [bottom_up_features[f] for f in self.in_features]
        # feature_map_visualization(x, 'resnet', 'feats_visualization_resnet_a3fpn', only_save_merged=False)
        results = self.a3fpn(results)

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)

        # feature_map_visualization(results[0], module_type='fpn_last_fusion',
        #                           save_dir="feats_visualization_with_resample_with_weight_with_assemble")
        # draw_heatmap(results[0], module_type='a3fpn_last_fusion',
        #              save_dir="heat_map_feats_visualization_with_resample_with_weight_with_assemble_0_pic3",
        #              )

        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for
                name in self._out_features}


@BACKBONE_REGISTRY.register()
def build_resnet_a3fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
        input_shape:

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.A3FPN.IN_FEATURES
    out_channels = cfg.MODEL.A3FPN.OUT_CHANNELS
    backbone = A3FPN_backbone(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(),
        squeeze=cfg.MODEL.A3FPN.SQUEEZE,
        act=nn.GELU(),
        norm=cfg.MODEL.A3FPN.NORM,
        compress_channel=cfg.MODEL.A3FPN.COMPRESS_CHANNELS,
        group_num=cfg.MODEL.A3FPN.GROUP_NUMS,
        num_repblocks=cfg.MODEL.A3FPN.NUM_REPBLOCKS,
        expansion=cfg.MODEL.A3FPN.EXPANSION,
        using_resampling=cfg.MODEL.A3FPN.USING_RESAMPLING,
        dcn_groups=cfg.MODEL.A3FPN.DCN_GROUPS,
        dcn_config=cfg.MODEL.A3FPN.DCN_CONFIG,
    )
    return backbone
