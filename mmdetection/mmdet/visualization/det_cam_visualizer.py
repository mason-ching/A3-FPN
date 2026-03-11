# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy
import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from .collate import collate
from .scatter_gather import scatter
from typing import List, Optional, Union, Tuple
from mmengine.config import Config, ConfigDict
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector

try:
    from pytorch_grad_cam import (AblationCAM, AblationLayer,
                                  ActivationsAndGradients)
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')
from mmcv.transforms import Compose
import warnings
from mmdet.utils import ConfigType
from mmdet.registry import DATASETS, MODELS
from mmengine.registry import DefaultScope
from ..evaluation import get_classes
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.model.utils import revert_sync_batchnorm


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def reshape_transform(feats, max_shape=(20, 20), is_need_grad=False):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.

    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if len(max_shape) == 1:
        max_shape = max_shape * 2

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    else:
        if is_need_grad:
            raise NotImplementedError('The `grad_base` method does not '
                                      'support output multi-activation layers')

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, args, device='cuda:0', scope='mmdet'):
        super().__init__()
        if scope is None:
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope = default_scope.scope_name
        init_default_scope(scope)

        self.cfg = Config.fromfile(cfg)
        if args.cfg_options:
            self.cfg.merge_from_dict(args.cfg_options)

        self.device = device
        self.score_thr = args.score_thr
        self.checkpoint = args.checkpoint
        # self.detector = self._init_model(cfg, self.checkpoint, device)
        # self.detector = revert_sync_batchnorm(self.detector)
        self.detector = self.build_detector()

        self.return_loss = False
        self.input_data = None
        self.img = None

    def build_detector(self):
        detector = init_detector(self.cfg, self.checkpoint, device=self.device)
        return detector

    def _init_model(
            self,
            cfg: ConfigType,
            weights: Optional[str],
            device: str = 'cpu',
    ) -> nn.Module:
        """Initialize the model with the given config and checkpoint on the
        specific device.

        Args:
            cfg (ConfigType): Config containing the model information.
            weights (str, optional): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        checkpoint: Optional[dict] = None
        if weights is not None:
            checkpoint = _load_checkpoint(weights, map_location='cpu')

        if not cfg:
            assert checkpoint is not None
            try:
                # Prefer to get config from `message_hub` since `message_hub`
                # is a more stable module to store all runtime information.
                # However, the early version of MMEngine will not save config
                # in `message_hub`, so we will try to load config from `meta`.
                cfg_string = checkpoint['message_hub']['runtime_info']['cfg']
            except KeyError:
                assert 'meta' in checkpoint, (
                    'If model(config) is not provided, the checkpoint must'
                    'contain the config string in `meta` or `message_hub`, '
                    'but both `meta` and `message_hub` are not found in the '
                    'checkpoint.')
                meta = checkpoint['meta']
                if 'cfg' in meta:
                    cfg_string = meta['cfg']
                else:
                    raise ValueError(
                        'Cannot find the config in the checkpoint.')
            cfg.update(
                Config.fromstring(cfg_string, file_format='.py')._cfg_dict)

        # Delete the `pretrained` field to prevent model from loading the
        # the pretrained weights unnecessarily.
        if cfg.model.get('pretrained') is not None:
            del cfg.model.pretrained

        model = MODELS.build(cfg.model)
        model.cfg = cfg
        self._load_weights_to_model(model, checkpoint, cfg)
        model.to(device)
        model.eval()
        return model

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v
                    for k, v in checkpoint_meta['dataset_meta'].items()
                }
            elif 'CLASSES' in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta['CLASSES']
                model.dataset_meta = {'classes': classes}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use COCO classes by default.')
                model.dataset_meta = {'classes': get_classes('coco')}
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn('weights is None, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

        # Priority:  args.palette -> config -> checkpoint
        test_dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def set_input_data(self, img, bboxes=None, labels=None):
        self.img = img
        cfg = copy.deepcopy(self.cfg)
        if self.return_loss:
            assert bboxes is not None
            assert labels is not None
            cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
            cfg.test_dataloader.dataset.pipeline = replace_ImageToTensor(
                cfg.test_dataloader.dataset.pipeline)
            cfg.test_dataloader.dataset.pipeline[1].transforms[-1] = dict(
                type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
            # TODO: support mask
            data = dict(
                img=self.img,
                gt_bboxes=bboxes,
                gt_labels=labels.astype(np.long),
                bbox_fields=['gt_bboxes'])
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0][0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            data['gt_bboxes'] = [
                gt_bboxes.data[0] for gt_bboxes in data['gt_bboxes']
            ]
            data['gt_labels'] = [
                gt_labels.data[0] for gt_labels in data['gt_labels']
            ]
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]

            data['img'] = data['img'][0]
            data['gt_bboxes'] = data['gt_bboxes'][0]
            data['gt_labels'] = data['gt_labels'][0]
        else:
            # set loading pipeline type
            cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
            # print(self.img.shape)
            data = dict(img=self.img)
            # cfg.test_dataloader.dataset.pipeline = replace_ImageToTensor(
            #     cfg.test_dataloader.dataset.pipeline)
            pipeline = cfg.test_dataloader.dataset.pipeline
            del pipeline[-2]  # 删除索引1的元素

            test_pipeline = Compose(pipeline)
            data = test_pipeline(data)
            # print(data['inputs'].permute(1, 2, 0).numpy().shape)
            # cv2.imwrite('picture.png', data['inputs'].permute(1, 2, 0).numpy())
            # print(data)
            # data = collate([data], samples_per_gpu=1)
            # # just get the actual data from DataContainer
            # data['img_metas'] = [
            #     img_metas.data[0] for img_metas in data['img_metas']
            # ]
            # data['img'] = [img.data[0] for img in data['img']]

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                # data = scatter(data, [self.device])[0]
                self.input_data = (data['inputs'].float().unsqueeze(0).to(self.device), [data['data_samples']])
            else:
                for m in self.detector.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'

                self.input_data = (torch.FloatTensor(data['inputs']).unsqueeze(0), [data['data_samples']])

        # self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.return_loss:
            loss = self.detector.loss(*self.input_data)
            return [loss]
        else:
            with torch.no_grad():
                # print(self.input_data)
                results = self.detector.predict(self.input_data[0], self.input_data[1])[0]

                # if isinstance(results, SampleList):
                print(results.get('pred_instances').labels)
                if 'bboxes' in results:
                    bbox_result = results['bboxes']
                if 'masks' in results:
                    segm_result = results['masks']
                else:
                    segm_result = None

                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(results['labels'])
                ]
                labels = np.concatenate(labels)

                segms = None
                if segm_result is not None and len(labels) > 0:  # non empty
                    segms = mmcv.concat_list(segm_result)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(
                            segms, dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                if self.score_thr > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > self.score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                    if segms is not None:
                        segms = segms[inds, ...]
                return [{'bboxes': bboxes, 'labels': labels, 'segms': segms}]


class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                         input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.

        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.

    Args:
        method:  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
    """

    def __init__(self,
                 method_class,
                 model,
                 target_layers,
                 reshape_transform=None,
                 is_need_grad=False,
                 extra_params=None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == 'AblationCAM':
            batch_size = extra_params.get('batch_size', 1)
            ratio_channels_to_ablate = extra_params.get(
                'ratio_channels_to_ablate', 1.)
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params['ablation_layer'],
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        else:
            self.cam = method_class(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        self.classes = model.detector.dataset_meta['classes']
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def switch_activations_and_grads(self, model):
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                         x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.

    If there is not a large enough overlap, or the category changed,
    assign a score of 0.

    The total score is the sum of all the box scores.
    """

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        output = torch.tensor([0.], device=self.device)

        if 'loss_cls' in results:
            # grad_base_method
            for loss_key, loss_value in results.items():
                if 'loss' not in loss_key:
                    continue
                if isinstance(loss_value, list):
                    output += sum(loss_value)
                else:
                    output += loss_value
            return output
        else:
            # grad_free_method
            if len(results['bboxes']) == 0:
                return output

            pred_bboxes = torch.from_numpy(results['bboxes']).to(self.device)
            pred_labels = results['labels']
            pred_segms = results['segms']

            if pred_segms is not None:
                pred_segms = torch.from_numpy(pred_segms).to(self.device)

            for focal_box, focal_label, focal_segm in zip(
                    self.focal_bboxes, self.focal_labels, self.focal_segms):
                ious = torchvision.ops.box_iou(focal_box[None],
                                               pred_bboxes[..., :4])
                index = ious.argmax()
                if ious[0, index] > self.match_iou_thr and pred_labels[
                    index] == focal_label:
                    # TODO: Adaptive adjustment of weights based on algorithms
                    score = ious[0, index] + pred_bboxes[..., 4][index]
                    output = output + score

                    if focal_segm is not None and pred_segms is not None:
                        segms_score = (focal_segm *
                                       pred_segms[index]).sum() / (
                                              focal_segm.sum() +
                                              pred_segms[index].sum() + 1e-7)
                        output = output + segms_score
            return output


# TODO: Fix RuntimeError: element 0 of tensors does not require grad and
#  does not have a grad_fn.
#  Can be removed once the source code is fixed.
class EigenCAM(BaseCAM):

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            uses_gradients=False)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return get_2d_projection(activations)


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.

    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)
