_base_ = [
    '../_base_/models/ssd300.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Dataset Settings
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/JHU-CROWD++-2/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=[123.675, 116.28, 103.53],
                to_rgb=True,
                ratio_range=(1, 4)
            ),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3
            ),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18
            ),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/kaggle/working/JHU-CROWD++-2/valid/_annotations.coco.json',
    metric='bbox'
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='PackDetInputs')
        ]
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='/kaggle/working/JHU-CROWD++-2/test/_annotations.coco.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./work_dirs/coco_detection/test'
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=5e-4)
)

auto_scale_lr = dict(base_batch_size=64)
