_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)
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
        contract_dilation=True,
        # type='ResNet',
        # depth=50,
        # deep_stem=False,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=-1,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        # # norm_eval=True,
        # style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    decode_head=
    dict(
        type='A3FPNUperNetHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        squeeze=[1, 2, 4, 1],
        norm_cfg=norm_cfg,
        # act_cfg=dict(type='GELU'),
        align_corners=False,
        compress_channel=[16, 16, 16, 32],
        group_num=[16, 16, 16, 32],
        num_repblocks=1,
        expansion=4.,
        head_using_resampling=False,
        a3fpn_using_resampling=[True, True, True],
        dcn_groups=[16, 16, 16, 32],
        dcn_config=dict(
            dcn_norm='LN',
            offset_scale=2.0,
            dw_kernel_size=3,
            dcn_output_bias=True,
            center_feature_scale=False,
            remove_center=False,
            without_pointwise=False,
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        # act_cfg=dict(type='GELU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    ),
    # model training and testing settings
    train_cfg=dict(
        # num_points=2048, oversample_ratio=3, importance_sample_ratio=0.75
    ),
    test_cfg=dict(
        mode='whole',
        # subdivision_steps=2,
        # subdivision_num_points=8196,
        # scale_factor=2
    ),
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/CityScapes'
# crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ]
    )
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
        'mDice', 'mFscore'
    ]
)
test_evaluator = val_evaluator

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005,
    # type='AdamW',
    # lr=0.00006,
    # betas=(0.9, 0.999),
    # weight_decay=0.01,
)

optim_wrapper = dict(
    type='OptimWrapper',
    # _delete_=True,
    # type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    # paramwise_cfg={
    #     'decay_rate': 1.0,
    #     'decay_type': 'layer_wise',
    #     'layers': [2, 2, 4, 2],
    #     # 'num_subnet': 4,
    #     }
    # constructor='LearningRateDecayOptimizerConstructor',
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=4000
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=2.0,
        begin=4000,
        end=80000,
        by_epoch=False
    ),
]

# training schedule for 80k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=120000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200, save_best='mIoU', max_keep_ckpts=3,
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)

# # fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
# fp16 = dict()
# find_unused_parameters=True