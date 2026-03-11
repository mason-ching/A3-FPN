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
from detectron2.layers import (ShapeSpec, get_norm)

__all__ = ["build_resnet_prfpn_backbone", "build_retinanet_resnet_prfpn_backbone", "PRFPN_backbone", 'SampleBlock']


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
    default_act = nn.SiLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act: Union[bool, nn.Module] = True, bias=False,
                 norm: Union[bool, nn.Module] = 'SyncBN',
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


'''                                                        SCR                                              '''


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


class SRB(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True,
                 num_fusion=2,
                 compress_c=16):
        super().__init__()
        if num_fusion == 2:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 2, num_fusion, kernel_size=1, stride=1, padding=0)
        elif num_fusion == 3:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_3 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 3, num_fusion, kernel_size=1, stride=1, padding=0)
        else:
            self.weight_level_1 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_2 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_3 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_level_4 = Conv(oup_channels, compress_c, 1, 1)
            self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)

        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):

        if len(x) == 2:
            input1, input2 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        elif len(x) == 3:
            input1, input2, input3 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)
            level_3_weight_v = self.weight_level_3(input3)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :] + input3 * \
                                levels_weight[:, 2:, :, :]
        else:
            input1, input2, input3, input4 = x
            level_1_weight_v = self.weight_level_1(input1)
            level_2_weight_v = self.weight_level_2(input2)
            level_3_weight_v = self.weight_level_3(input3)
            level_4_weight_v = self.weight_level_4(input4)

            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            # levels_weight = F.softmax(levels_weight, dim=1)
            levels_weight = torch.sigmoid(levels_weight)

            fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                                input2 * levels_weight[:, 1:2, :, :] + \
                                input3 * levels_weight[:, 2:3, :, :] + \
                                input4 * levels_weight[:, 3:, :, :]

        gn_x = self.gn(fused_out_reduced)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1).contiguous()
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * fused_out_reduced
        x_2 = w2 * fused_out_reduced
        y = self.restructure(x_1, x_2)
        return y

    @staticmethod
    def restructure(x_1, x_2):
        return x_1 + x_2.flip(1).contiguous()


class CRB(nn.Module):
    """
    gamma: 0<gamma<1
    """

    def __init__(self,
                 op_channel: int,
                 gamma: float = 1 / 2,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()

        self.low_channel = low_channel = int(gamma * op_channel)
        self.up_channel = up_channel = op_channel - low_channel
        self.squeeze1 = nn.Conv2d(low_channel, low_channel // squeeze_ratio, kernel_size=1, bias=False) \
            if squeeze_ratio > 1 else nn.Identity()
        self.squeeze2 = nn.Conv2d(up_channel, up_channel // squeeze_ratio, kernel_size=1, bias=False) \
            if squeeze_ratio > 1 else nn.Identity()
        # lower
        self.GWC = nn.Conv2d(low_channel // squeeze_ratio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(low_channel // squeeze_ratio, op_channel, kernel_size=1, bias=False)

        # upper
        self.PWC2 = nn.Conv2d(up_channel // squeeze_ratio, op_channel - (up_channel // squeeze_ratio), kernel_size=1,
                              bias=False)

        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # partition
        low, up = torch.split(x, [self.low_channel, self.up_channel], dim=1)
        low, up = self.squeeze1(low), self.squeeze2(up)
        # re-extract
        Z1 = self.GWC(low) + self.PWC1(low)
        Z2 = torch.cat([self.PWC2(up), up], dim=1)
        # re-fuse
        out = torch.cat([Z1, Z2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCR(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 12,
                 group_kernel_size: int = 3,
                 gate_treshold: float = 0.5,
                 gamma: float = 1 / 2,
                 squeeze_ratio: int = 1,
                 group_size: int = 2,
                 num_fusion=2,
                 compress_c=16
                 ):
        super().__init__()
        self.SRB = SRB(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold,
                       num_fusion=num_fusion,
                       compress_c=compress_c)
        # self.CRB = CRB(op_channel,
        #                gamma=gamma,
        #                squeeze_ratio=squeeze_ratio,
        #                group_size=group_size,
        #                group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRB(x)
        # x = self.CRB(x)
        return x


'''                                             PR-FPN                                            '''


class SampleBlock(nn.Module):
    """
    semantic-guided sampling
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            unsampled_nc: int,
            sample: str = 'upsample',
            scale_factor: int = 2,
            upsample_method: str = 'bilinear',
            align_corners: Optional[bool] = False,
            group: int = 1,
            act: Optional[Union[bool, nn.Module]] = nn.SiLU(inplace=True),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            # dcn arguments
            using_offset: Optional[bool] = False,
            dcn_group: int = 4,
            dcn_config: dict = dict(
                dcn_norm='LN',
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            )
    ) -> None:
        """
        Args:
            in_channels:
            out_channels:
            unsampled_nc:
            sample:
            scale_factor:
            upsample_method:
            align_corners:
            group:
            act:
            norm:
            using_offset:
            dcn_group:
            dcn_config:
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

        # self.using_offset = using_offset
        if using_offset:
            self.dcn_layer = DCN(
                in_nc=out_channels + unsampled_nc,
                sampled_out_nc=out_channels,
                act=act,
                dcn_group=dcn_group,
                **dcn_config
            )
        else:
            self.dcn_layer = None

    def forward(self, feat_u: Tensor, feat_s: Tensor) -> Tensor:
        """
        Args:
            feat_u (Tensor): unsampled feature maps that guide the sampling
            feat_s (Tensor): feature maps that need to be sampled
        return: Tensor
        """
        sampled_feat = self.sample_layer(feat_s)
        # if sampled_feat.shape[2:] != feat_u.shape[2:]:
        #     sampled_feat = resize(sampled_feat, feat_u.shape[2:])
        adjusted_features = self.dcn_layer(feat_u.float(), sampled_feat.float()) if self.dcn_layer else sampled_feat
        return adjusted_features


class DCN(nn.Module):
    """
    spatially guide sampled feature maps by deformable convolutions
    """

    def __init__(
            self,
            in_nc: int,
            sampled_out_nc: int = 128,
            act: Optional[Union[bool, nn.Module]] = True,
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
             in_nc:
             sampled_out_nc:
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
        super(DCN, self).__init__()
        self.offset = Conv(in_nc, sampled_out_nc, k=1, s=1, p=0, bias=False, act=act, norm='SyncBN')
        self.dcn = dcn_v4(
            sampled_out_nc, 3, stride=1, pad=1, dilation=1, group=dcn_group, offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            output_bias=dcn_output_bias,
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,
            without_pointwise=without_pointwise,
            extra_offset_mask=True
        )
        # self.dcn = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.norm = get_norm(dcn_norm, sampled_out_nc) if dcn_norm else nn.Identity()

    def forward(self, feat_u: Tensor, sampled_feat: Tensor) -> Tensor:
        """
        Args:
            feat_u (Tensor): unsampled feature maps that guide the spatial adjustment
            sampled_feat (Tensor): feature maps that have been sampled
        return: Tensor
        """
        offset = self.offset(
            torch.cat([feat_u, sampled_feat], dim=1)  # concat for offset by computing the dif
        )
        adjusted_features = self.norm(
            self.dcn([sampled_feat, offset])  # [sampled_feat, offset]
        )
        return adjusted_features


class PRFPN_2(nn.Module):
    def __init__(
            self,
            inter_dim=512,
            level=0,
            channel=None,
            act=nn.SiLU(inplace=True),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            # dcn arguments
            using_offset: Optional[bool] = False,
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
        super(PRFPN_2, self).__init__()

        if channel is None:
            channel = [64, 128]
        self.inter_dim = inter_dim
        compress_c = 16
        self.level = level
        if self.level == 0:
            self.upsample = SampleBlock(
                channel[1], channel[0], channel[0], sample='upsample', scale_factor=2, act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config
            )
        else:
            self.downsample = SampleBlock(
                channel[0], channel[1], channel[1], sample='downsample', scale_factor=2, act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config
            )

        self.refine = SCR(self.inter_dim, 16, num_fusion=2, compress_c=compress_c)

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input1, input2)
        elif self.level == 1:
            input1 = self.downsample(input2, input1)

        out = self.refine((input1, input2))

        return out


class PRFPN_3(nn.Module):
    def __init__(
            self,
            inter_dim=512,
            level=0,
            channel=None,
            act=nn.SiLU(inplace=True),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            # dcn arguments
            using_offset: Optional[bool] = False,
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
        super(PRFPN_3, self).__init__()

        if channel is None:
            channel = [64, 128, 256]
        self.inter_dim = inter_dim
        compress_c = 16

        self.refine = SCR(self.inter_dim, 16, num_fusion=3, compress_c=compress_c)
        # else:
        #     self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(
                channel[2], channel[0], channel[0], sample='upsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        elif self.level == 2:
            self.downsample2x = SampleBlock(
                channel[1], channel[2], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample4x = SampleBlock(
                channel[0], channel[2], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input1, input2)
            input3 = self.upsample4x(input1, input3)

        elif self.level == 1:
            input3 = self.upsample2x1(input2, input3)
            input1 = self.downsample2x1(input2, input1)
        elif self.level == 2:
            input1 = self.downsample4x(input3, input1)
            input2 = self.downsample2x(input3, input2)

        out = self.refine((input1, input2, input3))
        # self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        return out


class PRFPN_4(nn.Module):
    def __init__(
            self,
            inter_dim=512,
            level=0,
            channel=None,
            act=nn.SiLU(inplace=True),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            # dcn arguments
            using_offset: Optional[bool] = False,
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
        super(PRFPN_4, self).__init__()
        if channel is None:
            channel = [256, 512, 1024, 2048]
        self.inter_dim = inter_dim
        compress_c = 16
        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(
                channel[2], channel[0], channel[0], sample='upsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.upsample2x = SampleBlock(
                channel[1], channel[0], channel[0], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.upsample8x = SampleBlock(
                channel[3], channel[0], channel[0], sample='upsample', scale_factor=8,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        elif self.level == 1:
            self.upsample2x1 = SampleBlock(
                channel[2], channel[1], channel[1], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.upsample4x1 = SampleBlock(
                channel[3], channel[1], channel[1], sample='upsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample2x1 = SampleBlock(
                channel[0], channel[1], channel[1], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        elif self.level == 2:
            self.upsample2x2 = SampleBlock(
                channel[3], channel[2], channel[2], sample='upsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample2x2 = SampleBlock(
                channel[1], channel[2], channel[2], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample4x2 = SampleBlock(
                channel[0], channel[2], channel[2], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        elif self.level == 3:
            self.downsample2x3 = SampleBlock(
                channel[2], channel[3], channel[3], sample='downsample', scale_factor=2,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample4x3 = SampleBlock(
                channel[1], channel[3], channel[3], sample='downsample', scale_factor=4,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )
            self.downsample8x3 = SampleBlock(
                channel[0], channel[3], channel[3], sample='downsample', scale_factor=8,
                act=act,
                norm=norm,
                using_offset=using_offset,
                dcn_group=dcn_group,
                dcn_config=dcn_config,
            )

        self.refine = SCR(self.inter_dim, 16, num_fusion=4, compress_c=compress_c)

    def forward(self, x):
        input1, input2, input3, input4 = x

        if self.level == 0:
            input2 = self.upsample2x(input1, input2)
            input3 = self.upsample4x(input1, input3)
            input4 = self.upsample8x(input1, input4)

        elif self.level == 1:
            input1 = self.downsample2x1(input2, input1)
            input3 = self.upsample2x1(input2, input3)
            input4 = self.upsample4x1(input2, input4)

        elif self.level == 2:
            input1 = self.downsample4x2(input3, input1)
            input2 = self.downsample2x2(input3, input2)
            input4 = self.upsample2x2(input3, input4)

        elif self.level == 3:

            input1 = self.downsample8x3(input4, input1)
            input2 = self.downsample4x3(input4, input2)
            input3 = self.downsample2x3(input4, input3)

        out = self.refine((input1, input2, input3, input4))

        return out


class Body(nn.Module):
    def __init__(
            self,
            channels=None,
            act=nn.SiLU(inplace=True),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            extra_num_layers: int = 0,
            # dcn arguments
            using_offset: Union[List[bool], Tuple[bool]] = False,
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
        super(Body, self).__init__()

        if channels is None:
            channels = [64, 128, 256, 512]

        # top-down
        self.prfpn_2_level0 = PRFPN_2(
            inter_dim=channels[-2], level=0, channel=channels[-2:],
            act=act,
            norm=norm,
            using_offset=using_offset[0],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.prfpn_2_level1 = PRFPN_2(
            inter_dim=channels[-1], level=1, channel=channels[-2:],
            act=act,
            norm=norm,
            using_offset=using_offset[0],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )

        self.prfpn_3_level0 = PRFPN_3(
            inter_dim=channels[-3], level=0, channel=channels[-3:],
            act=act,
            norm=norm,
            using_offset=using_offset[1],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.prfpn_3_level1 = PRFPN_3(
            inter_dim=channels[-2], level=1, channel=channels[-3:],
            act=act,
            norm=norm,
            using_offset=using_offset[1],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )
        self.prfpn_3_level2 = PRFPN_3(
            inter_dim=channels[-1], level=2, channel=channels[-3:],
            act=act,
            norm=norm,
            using_offset=using_offset[1],
            dcn_group=dcn_groups[2],
            dcn_config=dcn_config,
        )

        self.prfpn_4_level0 = PRFPN_4(
            inter_dim=channels[-4], level=0, channel=channels[-4:],
            act=act,
            norm=norm,
            using_offset=using_offset[2],
            dcn_group=dcn_groups[0],
            dcn_config=dcn_config,
        )
        self.prfpn_4_level1 = PRFPN_4(
            inter_dim=channels[-3], level=1, channel=channels[-4:],
            act=act,
            norm=norm,
            using_offset=using_offset[2],
            dcn_group=dcn_groups[1],
            dcn_config=dcn_config,
        )
        self.prfpn_4_level2 = PRFPN_4(
            inter_dim=channels[-2], level=2, channel=channels[-4:],
            act=act,
            norm=norm,
            using_offset=using_offset[2],
            dcn_group=dcn_groups[2],
            dcn_config=dcn_config,
        )
        self.prfpn_4_level3 = PRFPN_4(
            inter_dim=channels[-1], level=3, channel=channels[-4:],
            act=act,
            norm=norm,
            using_offset=using_offset[2],
            dcn_group=dcn_groups[3],
            dcn_config=dcn_config,
        )

        self.extra_layers = nn.ModuleList()
        for i in range(extra_num_layers):
            self.extra_layers.append(
                nn.ModuleList(
                    [
                        PRFPN_4(
                            inter_dim=channels[-4], level=0, channel=channels[-4:],
                            act=act,
                            norm=norm,
                            using_offset=using_offset[3 + i],
                            dcn_group=dcn_groups[0],
                            dcn_config=dcn_config,
                        ),
                        PRFPN_4(
                            inter_dim=channels[-3], level=1, channel=channels[-4:],
                            act=act,
                            norm=norm,
                            using_offset=using_offset[3 + i],
                            dcn_group=dcn_groups[1],
                            dcn_config=dcn_config,
                        ),
                        PRFPN_4(
                            inter_dim=channels[-2], level=2, channel=channels[-4:],
                            act=act,
                            norm=norm,
                            using_offset=using_offset[3 + i],
                            dcn_group=dcn_groups[2],
                            dcn_config=dcn_config,
                        ),
                        PRFPN_4(
                            inter_dim=channels[-1], level=3, channel=channels[-4:],
                            act=act,
                            norm=norm,
                            using_offset=using_offset[3 + i],
                            dcn_group=dcn_groups[3],
                            dcn_config=dcn_config,
                        ),
                    ]
                )
            )

        # bottom-up
        # self.prfpn_2_level0 = PRFPN_2(
        #     inter_dim=channels[0], level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[0],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_2_level1 = PRFPN_2(
        #     inter_dim=channels[1], level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[0],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        #
        # self.prfpn_3_level0 = PRFPN_3(
        #     inter_dim=channels[0], level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[1],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_3_level1 = PRFPN_3(
        #     inter_dim=channels[1], level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[1],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_3_level2 = PRFPN_3(
        #     inter_dim=channels[2], level=2, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[1],
        #     dcn_group=dcn_groups[2],
        #     dcn_config=dcn_config,
        # )
        #
        # self.prfpn_4_level0 = PRFPN_4(
        #     inter_dim=channels[0], level=0, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[2],
        #     dcn_group=dcn_groups[0],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_4_level1 = PRFPN_4(
        #     inter_dim=channels[1], level=1, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[2],
        #     dcn_group=dcn_groups[1],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_4_level2 = PRFPN_4(
        #     inter_dim=channels[2], level=2, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[2],
        #     dcn_group=dcn_groups[2],
        #     dcn_config=dcn_config,
        # )
        # self.prfpn_4_level3 = PRFPN_4(
        #     inter_dim=channels[3], level=3, channel=channels,
        #     act=act,
        #     norm=norm,
        #     using_offset=using_offset[2],
        #     dcn_group=dcn_groups[3],
        #     dcn_config=dcn_config,
        # )
        #
        # self.extra_layers = nn.ModuleList()
        # for i in range(extra_num_layers):
        #     self.extra_layers.append(
        #         nn.ModuleList(
        #             [
        #                 PRFPN_4(
        #                     inter_dim=channels[0], level=0, channel=channels,
        #                     act=act,
        #                     norm=norm,
        #                     using_offset=using_offset[3 + i],
        #                     dcn_group=dcn_groups[0],
        #                     dcn_config=dcn_config,
        #                 ),
        #                 PRFPN_4(
        #                     inter_dim=channels[1], level=1, channel=channels,
        #                     act=act,
        #                     norm=norm,
        #                     using_offset=using_offset[3 + i],
        #                     dcn_group=dcn_groups[1],
        #                     dcn_config=dcn_config,
        #                 ),
        #                 PRFPN_4(
        #                     inter_dim=channels[2], level=2, channel=channels,
        #                     act=act,
        #                     norm=norm,
        #                     using_offset=using_offset[3 + i],
        #                     dcn_group=dcn_groups[2],
        #                     dcn_config=dcn_config,
        #                 ),
        #                 PRFPN_4(
        #                     inter_dim=channels[3], level=3, channel=channels,
        #                     act=act,
        #                     norm=norm,
        #                     using_offset=using_offset[3 + i],
        #                     dcn_group=dcn_groups[3],
        #                     dcn_config=dcn_config,
        #                 ),
        #             ]
        #         )
        #     )

    def forward(self, x):
        x0, x1, x2, x3 = x

        # bottom-up
        # output0 = self.prfpn_2_level0((x0, x1))
        # x1 = self.prfpn_2_level1((x0, x1))
        #
        # x0 = self.prfpn_3_level0((output0, x1, x2))
        # output1 = self.prfpn_3_level1((output0, x1, x2))
        # x2 = self.prfpn_3_level2((output0, x1, x2))
        #
        # output0 = self.prfpn_4_level0((x0, output1, x2, x3))
        # x1 = self.prfpn_4_level1((x0, output1, x2, x3))
        # output2 = self.prfpn_4_level2((x0, output1, x2, x3))
        # x3 = self.prfpn_4_level3((x0, output1, x2, x3))
        #
        # outputs = [output0, x1, output2, x3]
        # for extra_layer in self.extra_layers:
        #     prfpn_4_level0, prfpn_4_level1, prfpn_4_level2, prfpn_4_level3 = extra_layer
        #     input0, input1, input2, input3 = outputs
        #     outputs[0] = prfpn_4_level0((input0, input1, input2, input3))
        #     outputs[1] = prfpn_4_level1((input0, input1, input2, input3))
        #     outputs[2] = prfpn_4_level2((input0, input1, input2, input3))
        #     outputs[3] = prfpn_4_level3((input0, input1, input2, input3))

        # top-down
        output2 = self.prfpn_2_level0((x2, x3))
        x3 = self.prfpn_2_level1((x2, x3))

        output1 = self.prfpn_3_level0((x1, output2, x3))
        x2 = self.prfpn_3_level1((x1, output2, x3))
        x3 = self.prfpn_3_level2((x1, output2, x3))

        output0 = self.prfpn_4_level0((x0, output1, x2, x3))
        x1 = self.prfpn_4_level1((x0, output1, x2, x3))
        output2 = self.prfpn_4_level2((x0, output1, x2, x3))
        x3 = self.prfpn_4_level3((x0, output1, x2, x3))

        outputs = [output0, x1, output2, x3]
        for extra_layer in self.extra_layers:
            prfpn_4_level0, prfpn_4_level1, prfpn_4_level2, prfpn_4_level3 = extra_layer
            input0, input1, input2, input3 = outputs
            outputs[0] = prfpn_4_level0((input0, input1, input2, input3))
            outputs[1] = prfpn_4_level1((input0, input1, input2, input3))
            outputs[2] = prfpn_4_level2((input0, input1, input2, input3))
            outputs[3] = prfpn_4_level3((input0, input1, input2, input3))

        return outputs


class PRFPN(nn.Module):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            squeeze: Union[List[int], int] = 1,
            act=nn.GELU(),
            norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
            extra_num_layers: int = 0,
            # dcn arguments
            using_offset: Union[List[bool], Tuple[bool]] = False,
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
        super(PRFPN, self).__init__()

        if isinstance(squeeze, int):
            squeeze = [squeeze] * len(in_channels)
        if isinstance(dcn_groups, int):
            dcn_groups = [dcn_groups] * len(in_channels)

        in_channels_reduced = [channel // squeeze[i] for i, channel in enumerate(in_channels)]
        # self.fp16_enabled = False

        self.conv0 = Conv(in_channels[0], in_channels_reduced[0], 1, act=act, norm=norm) \
            if squeeze[0] != 1 else nn.Identity()
        self.conv1 = Conv(in_channels[1], in_channels_reduced[1], 1, act=act, norm=norm) \
            if squeeze[1] != 1 else nn.Identity()
        self.conv2 = Conv(in_channels[2], in_channels_reduced[2], 1, act=act, norm=norm) \
            if squeeze[2] != 1 else nn.Identity()
        self.conv3 = Conv(in_channels[3], in_channels_reduced[3], 1, act=act, norm=norm) \
            if squeeze[3] != 1 else nn.Identity()

        self.prfpn_body = Body(
            channels=in_channels_reduced,
            act=act,
            norm=norm,
            extra_num_layers=extra_num_layers,
            using_offset=using_offset,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )

        self.conv00 = Conv(in_channels_reduced[0], out_channels, 3, p=1, act=act, norm=norm) \
            if in_channels_reduced[0] != out_channels else nn.Identity()
        self.conv11 = Conv(in_channels_reduced[1], out_channels, 3, p=1, act=act, norm=norm) \
            if in_channels_reduced[1] != out_channels else nn.Identity()
        self.conv22 = Conv(in_channels_reduced[2], out_channels, 3, p=1, act=act, norm=norm) \
            if in_channels_reduced[2] != out_channels else nn.Identity()
        self.conv33 = Conv(in_channels_reduced[3], out_channels, 3, p=1, act=act, norm=norm) \
            if in_channels_reduced[3] != out_channels else nn.Identity()

        # init weight
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

        out0, out1, out2, out3 = self.prfpn_body((x0, x1, x2, x3))

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


class PRFPN_backbone(Backbone):
    """
    This module implements :paper:`PR-FPN`.
    """

    def __init__(
                 self,
                 bottom_up,
                 in_features,
                 out_channels,
                 top_block=None,
                 squeeze=[1, 2, 2, 4],
                 act=nn.GELU(),
                 norm: Optional[Union[bool, nn.Module, str]] = 'SyncBN',
                 extra_num_layers: int = 0,
                 # dcn arguments
                 using_offset: Union[List[bool], Tuple[bool]] = False,
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
                are used to generate PRFPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which PRFPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                PRFPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra PRFPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
        """
        super(PRFPN_backbone, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self.prfpn = PRFPN(
            in_channels=in_channels_per_feature,
            out_channels=out_channels,
            squeeze=squeeze,
            act=act,
            norm=norm,
            extra_num_layers=extra_num_layers,
            using_offset=using_offset,
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
                mapping from feature map name to PRFPN feature map tensor
                in high to low resolution order. Returned feature names follow the PRFPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features]
        # feature_map_visualization(x, 'resnet', 'feats_visualization_resnet_prfpn', only_save_merged=False)
        results = self.prfpn((x[0], x[1], x[2], x[3]))
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # feature_map_visualization(results, 'prfpn', 'feats_visualization_prfpn', only_save_merged=False)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for
                name in self._out_features}


@BACKBONE_REGISTRY.register()
def build_resnet_prfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
        input_shape:

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.PRFPN.IN_FEATURES
    out_channels = cfg.MODEL.PRFPN.OUT_CHANNELS
    backbone = PRFPN_backbone(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(),
        squeeze=cfg.MODEL.PRFPN.SQUEEZE,
        act=nn.GELU(),
        norm=cfg.MODEL.PRFPN.NORM,
        extra_num_layers=cfg.MODEL.PRFPN.EXTRA_NUM_LAYERS,
        using_offset=cfg.MODEL.PRFPN.USING_OFFSET,
        dcn_groups=cfg.MODEL.PRFPN.DCN_GROUPS,
        dcn_config=cfg.MODEL.PRFPN.DCN_CONFIG,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_prfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
        input_shape

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.PRFPN.IN_FEATURES
    out_channels = cfg.MODEL.PRFPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = PRFPN_backbone(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        squeeze=cfg.MODEL.PRFPN.SQUEEZE,
        act=nn.GELU(),
        norm=cfg.MODEL.PRFPN.NORM,
        extra_num_layers=cfg.MODEL.PRFPN.EXTRA_NUM_LAYERS,
        using_offset=cfg.MODEL.PRFPN.USING_OFFSET,
        dcn_groups=cfg.MODEL.PRFPN.DCN_GROUPS,
        dcn_config=cfg.MODEL.PRFPN.DCN_CONFIG,
    )
    return backbone


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


def feature_map_visualization(
        x: Union[Tensor, List[Tensor], Tuple[Tensor]],
        module_type: str,
        save_dir: str = "feats_visualization",
        using_interpolation: bool = False,
        only_save_merged: bool = False,
        scales: Union[list, tuple] = (4, 8, 16, 32, 64), ) -> None:
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
        feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
        for i in range(0, channels):

            out_path = save_dir / f"output{j + 1}/{module_type.split('.')[-1]}_features_channel{i + 1}.png"  # out_path

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            feature_map_split = feature_map[i, :, :]
            feature_map_split = np.expand_dims(feature_map_split, axis=2)
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

            plt.imshow(feature_map_split)
            LOGGER.info(f"Saving {out_path}... ({i + 1}/{channels})")

            plt.axis('off')
            plt.savefig(out_path, dpi=80)
            plt.close()

            # np.save(str(out_path.with_suffix(".npy")), feature_map_split)  # npy save

        out_path = save_dir / f"output{j + 1}/{module_type.split('.')[-1]}_features_merged-channel.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        feature_map_sum = BI(feature_map_sum)

        # feature_map_sum = feature_map_sum.astype(np.uint8)
        # imsave(out_path, feature_map_sum, vmin=0, vmax=255)

        # cv2.imwrite(out_path, convert_grayscale_to_color(feature_map_sum, colormap))

        # img = Image.fromarray(feature_map_sum)
        # img = img.convert('RGB')
        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(2)
        # img.save(out_path)

        plt.imshow(feature_map_sum)
        plt.axis('off')
        LOGGER.info(f"Saving {out_path}... (channel-merged feature map)")
        plt.savefig(out_path, dpi=80)
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
