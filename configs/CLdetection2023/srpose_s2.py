default_scope = 'mmpose'
specified_class =None
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='SDR 2.0mm',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

backend_args = dict(backend='local')

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)

codec = dict(
    type='MSRAHeatmap',
    input_size=(1024, 1024),
    heatmap_size=(1024, 1024),
    # input_size=(960, 960),
    # heatmap_size=(960, 960),
    sigma=6,
    unbiased=True)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[121.25,121.25,121.25],
        std=[76.5, 76.5, 76.5],
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                multiscale_output = True,
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'
        )),
    head = dict(
        type='SRPoseHead',
        in_channels=[384,192,96,48],
        out_channels=[384,192,96,48],
        num_joints=38,
        upsample_log=[3, 2, 1, 2],
        per_emb_nums=[16, 8, 4, 4],
        supervises=[False, False, True, True],
        loss=dict(type='JointsMSELoss', use_target_weight=True),
        decoder=codec),


    # head=dict(
    #     type='HeatmapHead',
    #     in_channels=48,
    #     out_channels=38,
    #     deconv_out_channels=(256,256),
    #     deconv_kernel_sizes = (2,2),
    #     # conv_out_channels
    #     # conv_kernel_sizes = (3,3,3),
    #     specified_target_class = specified_class,
    #     loss=dict(type='KeypointMSELoss', use_target_weight=True),
    #     decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False))


dataset_type = 'CephalometricDataset'
data_mode = 'topdown'
data_root = './data/'


# meta_keys is used to add 'spacing' information, please do not change it if you don't know its usage
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
                                           'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                                           'flip_direction', 'flip_indices', 'raw_ann_info', 'spacing'))
]


val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
                                           'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                                           'flip_direction', 'flip_indices', 'raw_ann_info', 'spacing'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='train.json',
        data_prefix=dict(img='preprocessed/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='valid.json',
        data_prefix=dict(img='preprocessed/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='test.json',
        data_prefix=dict(img='preprocessed/'),
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='CephalometricMetric',
    target_class_id = specified_class)
test_evaluator = val_evaluator


