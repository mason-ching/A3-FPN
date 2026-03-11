# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN
from .prfpn import PRFPN_backbone, SampleBlock, build_resnet_prfpn_backbone, build_retinanet_resnet_prfpn_backbone
from .a3fpn import A3FPN_backbone, build_resnet_a3fpn_backbone, A3FPN
from .regnet import RegNet
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
)
from .vit import ViT, SimpleFeaturePyramid, get_vit_lr_decay_rate
from .mvit import MViT
from .swin import SwinTransformer
__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
