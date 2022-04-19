# dataset settings
dataset_type = 'CocoDataset'
bgtest_data_root = 'data'

obj_img_dir = 'data/Auto-retail-syndata-release/syn_image_train'
obj_mask_dir = 'data/Auto-retail-syndata-release/segmentation_labels'
shapenet_obj_img_dir = 'data/ShapeNetRendering'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
base_size = (800, 520)
albu_train_transforms = [
    dict(type="ShiftScaleRotate",
         shift_limit=0.1,
         scale_limit=0.1,
         rotate_limit=15,
         border_mode=0,
         value=0,
         p=0.3),
    dict(type="RandomBrightnessContrast",
         brightness_limit=[0.1, 0.6],
         contrast_limit=[0.1, 0.6],
         p=0.3),
    # dict(type="ImageCompression", quality_lower=85, quality_upper=95, p=0.3),
    dict(type="OneOf",
         transforms=[
              dict(type="Blur", blur_limit=8, p=1.0),
             #   dict(type="MedianBlur", blur_limit=3, p=1.0),
             #   dict(type="MotionBlur", blur_limit=80,  p=1.0)
         ],
         p=0.5),
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
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
    dict(type='Resize', img_scale=base_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type="Albu",
    #      transforms=albu_train_transforms,
    #      bbox_params=dict(
    #          type="BboxParams",
    #          format="pascal_voc",
    #          label_fields=["gt_labels"],
    #          min_visibility=0.0,
    #          filter_lost_elements=True),
    #      keymap={"img": "image", "gt_bboxes": "bboxes"},
    #      update_pad_shape=False,
    #      skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ClassAgnostic'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

testA_bg_train = dict(
    type=dataset_type,
    ann_file=bgtest_data_root + '/annotations/random_bg.json',
    img_prefix='data/random_bg',
    pipeline=train_pipeline)

testA_bg_train = dict(
    type='RepeatDataset',
    times=20,
    dataset=testA_bg_train)

# val and test from fix copy paste
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=testA_bg_train,
    val=dict(
        type='CocoAgnosticTest',
        classes=('object',),
        ann_file=bgtest_data_root + '/annotations/random_bg.json',
        img_prefix='data/random_bg',
        pipeline=test_pipeline),
    test=dict(
        type='CocoAgnosticTest',
        classes=('object',),
        ann_file=bgtest_data_root + '/annotations/random_bg.json',
        img_prefix='data/random_bg',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
