_base_ = "../mmcls/swin_transformer/swin-small_16xb64_in1k.py"
custom_imports = dict(imports=['cccls'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)
# model
load_from = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth"
model = dict(
    type="AICityImageClassifier",
    head=dict(
        type="AICityLinearClsHead",
        num_classes=116),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=116, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=116, prob=0.5)
    ]))
# data
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type="AICIty22Track4ClsDataset",
        data_prefix="./data/Auto-retail-syndata-release/syn_image_train/",
        ann_file="./data/Auto-retail-syndata-release/anns/train.txt"),
    val=dict(
        type="AICIty22Track4ClsDataset",
        data_prefix="./data/Auto-retail-syndata-release/syn_image_train/",
        ann_file="./data/Auto-retail-syndata-release/anns/val.txt"))
data['test'] = data['val']
# learning policy
runner = dict(max_epochs=10)
