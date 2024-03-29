img_scale = (640, 640)
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2),
    bbox_head=dict(type='YOLOXHead', num_classes=9, in_channels=192, feat_channels=192, with_faceKp=True, facezita_num_classes=11, with_facemohu=True),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.2, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/train0829.json',
            img_prefix='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            classes=('face', 'facewithmask', 'person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat')),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_visibles'])
        ]
    ),
    val=dict(
        type='CocoDataset',
        ann_file='/home/chase/shy/dataset/spjgh/train0829.json',
        img_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='/home/chase/shy/dataset/spjgh/train0829.json',
        img_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
max_epochs = 300
num_last_epochs = 15
interval = 10
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=30)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),
    dict(type='SyncNormHook', num_last_epochs=num_last_epochs, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=4)
evaluation = dict(
    save_best='auto', interval=1, dynamic_intervals=[(max_epochs-num_last_epochs, 1)], metric='bbox')
work_dir = './work_dirs/yolox_m_8x8_300e_coco2_kp_0109'
auto_resume = False
gpu_ids = [0,1]
find_unused_parameters=True
