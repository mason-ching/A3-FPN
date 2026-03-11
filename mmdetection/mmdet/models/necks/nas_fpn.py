# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.merge_cells import GlobalPoolingCell, SumCell
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


@MODELS.register_module()
class NASFPN(BaseModule):
    """NAS-FPN.

    Implementation of `NAS-FPN: Learning Scalable Feature Pyramid Architecture
    for Object Detection <https://arxiv.org/abs/1904.07392>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        stack_times: int,
        start_level: int = 0,
        end_level: int = -1,
        norm_cfg: OptConfigType = None,
        init_cfg: MultiConfig = dict(type='Caffe2Xavier', layer='Conv2d')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.stack_times = stack_times
        self.norm_cfg = norm_cfg

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(
                out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            self.extra_downsamples.append(
                nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))

        # add NAS FPN connections
        self.fpn_stages = ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

         Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build P6-P7 on top of P5
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        draw_heatmap(p4, module_type='nasfpn_last_fusion',
                     save_dir="heat_map_feats_visualization_pic2_level2",
                     )

        return p3, p4, p5, p6, p7

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
from typing import Union, List, Tuple, Optional
from torch import Tensor


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

    heatmap = np.maximum(feature_map, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_heatmap(
        featuremap: Union[Tensor, List[Tensor], Tuple[Tensor]],
        module_type: str,
        save_dir: str = "feats_visualization",
        img_path: str = '/home/mengen/Desktop/object_detection/009818.jpg',
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
        feature_map = torch.squeeze(feat, dim=0)
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
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)  # 将热力图的大小调整为与原始图像相同
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
        feature_map = torch.squeeze(feat, dim=0)
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

            plt.imshow(feature_map_split)
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

        plt.imshow(feature_map_sum)
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