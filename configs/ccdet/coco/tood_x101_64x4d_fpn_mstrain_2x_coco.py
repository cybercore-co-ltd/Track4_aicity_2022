_base_ = [
    '../../mmdet/tood/tood_x101_64x4d_fpn_mstrain_2x_coco.py',
]
load_from = "https://download.openmmlab.com/mmdetection/v2.0/tood/tood_x101_64x4d_fpn_mstrain_2x_coco/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth"
cudnn_benchmark = False
find_unused_parameters = True

# dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# AI City data
obj_img_dir = 'data/Auto-retail-syndata-release/syn_image_train'
obj_mask_dir = 'data/Auto-retail-syndata-release/segmentation_labels'
# ShapeNet data
shapenet_obj_img_dir = 'data/ShapeNetRendering'
base_size = (800, 520)
rescale = (0.8, 1.2)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomGenerateBGImage', width=1920, height=1080),
    dict(type='RandomGenerateBGImageGAN', width=1920,
         height=1080, gan_img_path=['data/CelebA_128x128_N2M2S64', 'data/LSUN_256x256_N2M2S128', 'data/biggan_imagenet']),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCopyPaste',
         min_area_ratio=0.02, max_area_ratio=0.3,
         img_foreground_dir=shapenet_obj_img_dir,
         max_paste_objets=6, min_paste_objects=2, center_paste=False),
    dict(type='CopyObjectsToBackgroundImage',
         min_area_ratio=0.002, max_area_ratio=0.1,
         img_foreground_dir=obj_img_dir, mask_foregroud_dir=obj_mask_dir,
         max_paste_objets=3, min_paste_objects=1,
         center_paste=True, with_mask=True),
    dict(type='Resize',
         img_scale=[
             (int(rescale[0]*base_size[0]), int(rescale[0]*base_size[1])),
             (int(rescale[1]*base_size[0]), int(rescale[1]*base_size[1]))],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ClassAgnostic', repeat_lable=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=[
         'img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=base_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

testA_bg_train = dict(
    type='CocoDataset',
    ann_file='data/annotations/random_bg.json',
    img_prefix='data/random_bg',
    pipeline=train_pipeline)

# val and test from fix copy paste
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=20,
        dataset=testA_bg_train),
    val=dict(
        type='CocoAgnosticTest',
        classes=('object',),
        ann_file='data/annotations/random_bg.json',
        img_prefix='data/random_bg',
        pipeline=test_pipeline),
    test=dict(
        type='CocoAgnosticTest',
        classes=('object',),
        ann_file='data/annotations/random_bg.json',
        img_prefix='data/random_bg',
        pipeline=test_pipeline))


# schedule
runner = dict(max_epochs=1)
# default runtime
custom_imports = dict(imports=['ccdet'], allow_failed_imports=False)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(_delete_=True, interval=1, metric='bbox')

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=0.1)}))
