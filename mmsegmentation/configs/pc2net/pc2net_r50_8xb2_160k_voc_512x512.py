_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/voc_leaf.py',
          ]
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32)
)

num_classes = 1
norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PC2MHeadv1',
        in_channels=[256, 512, 1024, 2048],
        level_output_channels=[256, 512, 1024, 2048],
        channels=256,
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        dropout_ratio=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=num_classes,
        # norm='',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # decode_head=dict(
    #     type='PC2MHeadv2',
    #     in_channels=[256, 512, 1024, 2048],
    #     level_output_channels=[256, 512, 1024, 2048],
    #     channels=512,
    #     in_index=(0, 1, 2, 3),
    #     input_transform='resize_concat',
    #     kernel_size=1,
    #     num_convs=1,
    #     concat_input=True,
    #     dropout_ratio=-1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# # dataset config
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(
#         type='RandomChoiceResize',
#         scales=[int(1024 * x * 0.1) for x in range(5, 21)],
#         resize_type='ResizeShortestEdge',
#         max_size=4096),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=dict(max_norm=0.01, norm_type=2))
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, save_best='mIoU', max_keep_ckpts=4,
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=2000))


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

