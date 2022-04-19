_base_ = "../mmcls/res2net/res2net50-w14-s8_8xb32_in1k.py"
custom_imports = dict(imports=['cccls'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)
# model
load_from = "https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth"
model = dict(
    type="AICityImageClassifier",
    head=dict(
        type="AICityLinearClsHead",
        num_classes=116))
# data
data = dict(
    samples_per_gpu=64,
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
lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=0)
runner = dict(max_epochs=10)
