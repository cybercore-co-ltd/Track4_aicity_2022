import argparse
import mmcv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='test config file path')
    parser.add_argument('--out', help='out file')
    args = parser.parse_args()
    return args


args = parse_args()

cfg = mmcv.Config.fromfile(args.cfg)

# dataset settings
base_size = cfg.data.test.pipeline[1].img_scale
rescale = (0.8, 1.2)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
if 'trident' in args.cfg:
    img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[
                        1.0, 1.0, 1.0], to_rgb=False)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(int(base_size[0]*rescale[0]), int(base_size[1]*rescale[0])),
                   (base_size[0], base_size[1]),
                   (int(base_size[0]*rescale[1]), int(base_size[1]*rescale[1]))],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg.data.val.pipeline = test_pipeline
cfg.data.test.pipeline = test_pipeline

cfg.dump(args.out)
