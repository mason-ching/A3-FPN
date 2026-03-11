"""Head in A3FPN"""
from typing import List, Tuple, Union, Optional, Any
import torch.nn as nn
from torch import Tensor
# import warnings
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList, OptMultiConfig
from .decode_head import BaseDecodeHead
import torch
from ..necks.a3fpn import SampleBlock, Conv, A3FPN, Resampler
from mmcv.cnn import ConvModule
from ..utils import resize
from .psp_head import PPM


class TransformBlockWithoutConv(nn.Module):
    """
    TransformBlockWithoutConv with args(in_channels, out_channel, activation, normalization.)
    that transforms hierarchical features (1/4, 1/8, 1/16, 1/32) into the lowest-level rich features.

    """

    def __init__(
            self,
            in_channels: list,
            act: Optional[Union[bool, nn.Module]] = nn.GELU(),
            norm_cfg: Optional[Union[bool, nn.Module, dict]] = dict(type='SyncBN', requires_grad=True),
            using_resampling=False,
            dcn_groups: Union[int, List[int], Union[int]] = 32,
            dcn_config: dict = dict(
                dcn_norm=dict(type='LN', requires_grad=True),
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
    ) -> None:
        """
            Initialize Conv layer with given arguments including normalization and activation.
            Args:
                in_channels (list): in channels
                act (bool or nn.Module): activation function
                using_resampling (bool):
            return: None
            """
        super().__init__()

        if isinstance(dcn_groups, int):
            dcn_groups = [dcn_groups] * len(in_channels)
        self.use_resampling = using_resampling
        self.in_channels = in_channels

        self.sample_blocks = nn.ModuleList(
            [
                SampleBlock(
                    in_channels=in_channels[i], out_channels=in_channels[i], sample='upsample',
                    scale_factor=2 ** i, act=act, align_corners=False,
                    upsample_method='bilinear',
                    norm_cfg=norm_cfg
                )
                for i in range(1, len(in_channels))
            ]
        )
        if self.use_resampling:
            self.context_conv = Conv(
                sum(in_channels), sum(in_channels[1:]),
                k=1, s=1, p=0, bias=True, act=act, norm_cfg=norm_cfg
            )
            self.resampler_blocks = nn.ModuleList(
                [
                    Resampler(
                        channels=in_channels[i],
                        act=act,
                        dcn_group=dcn_groups[i],
                        **dcn_config
                    )
                    for i in range(1, len(in_channels))
                ]
            )

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Apply sample blocks, condense block to input tensors."""
        for i, sample in enumerate(self.sample_blocks):
            i += 1
            x[i] = sample(x[i])
        if self.use_resampling:
            context_info = torch.split(
                self.context_conv(torch.cat(x, 1)), self.in_channels[1:], dim=1,
            )
            for i, resample in enumerate(self.resampler_blocks):
                i += 1
                x[i] = resample(context_info[i - 1], x[i])
        return x


@MODELS.register_module()
class A3FPNUperNetHead(BaseDecodeHead):
    """A3FPN in UperNet.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
            self,
            pool_scales=(1, 2, 3, 6),
            squeeze=(),
            norm_cfg: Optional[Union[bool, nn.Module, dict]] = dict(type='SyncBN', requires_grad=True),
            compress_channel: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            group_num: Union[List[int], Tuple[int]] = [16, 16, 32, 32],
            num_repblocks: int = 1,
            expansion: float = 1.,
            head_using_resampling=True,
            a3fpn_using_resampling: Union[List[bool], Tuple[bool]] = False,
            dcn_groups: Union[int, List[int], Union[int]] = 16,
            dcn_config: dict = dict(
                dcn_norm=dict(type='LN', requires_grad=True),
                offset_scale=0.5,
                dw_kernel_size=3,
                dcn_output_bias=False,
                center_feature_scale=False,
                remove_center=False,
                without_pointwise=False,
            ),
            **kwargs):
        super().__init__(input_transform='multiple_select', norm_cfg=norm_cfg, **kwargs)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        in_channels = self.in_channels[:-1]
        in_channels.append(self.channels)
        #
        self.a3fpn = A3FPN(
            in_channels,
            self.channels,
            squeeze=squeeze,
            act=nn.GELU(),
            norm_cfg=norm_cfg,
            compress_channel=compress_channel,
            group_num=group_num,
            num_repblocks=num_repblocks,
            expansion=expansion,
            using_resampling=a3fpn_using_resampling,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )

        self.transform_align = TransformBlockWithoutConv(
            in_channels=[self.channels] * len(in_channels),
            using_resampling=head_using_resampling, norm_cfg=norm_cfg, act=nn.GELU(),
            dcn_groups=32, dcn_config=dcn_config,
        ) if head_using_resampling else None
        # # FPN Module
        # self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
        #     l_conv = ConvModule(
        #         in_channels,
        #         self.channels,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     fpn_conv = ConvModule(
        #         self.channels,
        #         self.channels,
        #         3,
        #         padding=1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     self.lateral_convs.append(l_conv)
        #     self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        laterals = inputs[:-1]

        laterals.append(self.psp_forward(inputs))

        inputs = self.a3fpn(laterals)

        if self.transform_align:
            inputs = self.transform_align(inputs)
        else:
            for i in range(len(inputs) - 1, 0, -1):
                inputs[i] = resize(
                    inputs[i],
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        outs = torch.cat(inputs, dim=1)
        feats = self.fpn_bottleneck(outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
