_base_ = '../../mmdet/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
custom_imports = dict(imports=['ccdet'], allow_failed_imports=False)
# model
model = dict(
    type='FasterRCNN',
    roi_head=dict(mask_roi_extractor=None, mask_head=None),
    # model training and testing settings
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
# dataset settings
data = dict(
    val=dict(
        type='CocoAgnosticTest',
        classes=('object',),
        ann_file='data/annotations/random_bg.json',
        img_prefix='data/random_bg'))
data['test'] = data['val']
