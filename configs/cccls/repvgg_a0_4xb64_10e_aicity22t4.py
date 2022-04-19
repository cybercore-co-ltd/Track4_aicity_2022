_base_ = "../mmcls/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py"
custom_imports = dict(imports=['cccls'], allow_failed_imports=False)
# model
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth"
model = dict(
    type="AICityImageClassifier",
    head=dict(
        type="AICityLinearClsHead",
        num_classes=116))
# data
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
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
