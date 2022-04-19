import mmcv
import torch
import numpy as np

from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import collate, scatter

from mmcls.models import build_classifier
from mmcls.datasets.pipelines import Compose
from mmcv.runner import load_checkpoint, wrap_fp16_model


def init_classifier(config, checkpoint=None, device='cuda', fp16=False):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed classifier.
    """
    # Parse config
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    # Build model
    config.model.pretrained = None
    model = build_classifier(config.model)

    # Load checkpoint
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if ('meta' in checkpoint) and ('CLASSES' in checkpoint['meta']):
            model.CLASSES = checkpoint['meta']['CLASSES']

    # fuse backbone and conv-bn
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'fuse_model'):
        model.backbone.fuse_model()
    for m in model.modules():
        if hasattr(m, 'fuse_modules'):
            m.fuse_modules()
    model = fuse_conv_bn(model)

    # wrap_fp16_model
    if fp16:
        wrap_fp16_model(model)
        model.half()

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:

    def __call__(self, data):
        img = data['img']
        if isinstance(img, str):
            data['img_info'] = dict(filename=img)
            data['img_prefix'] = None
            img = mmcv.imread(data['img'])
            data['img'] = img
        return data


def build_cls_data_pipeline(test_pipeline):
    test_pipeline[0] = LoadImage()
    data_pipeline = Compose(test_pipeline)
    return data_pipeline


def get_cls_data(img, data_pipeline, device='cuda'):
    """Support both cpu and cuda device"""
    if isinstance(img, list):
        data = [data_pipeline(dict(img=item)) for item in img]
        if data[0]['img'][0].data.dtype == torch.uint8:
            for idx in range(len(data)):
                data[idx]['img'][0] = data[idx]['img'][0].float()
        data = collate(data, samples_per_gpu=len(data))
        if device == 'cuda':
            data = scatter(data, [device])[0]
    else:
        data = dict(img=img)
        data = data_pipeline(data)
        if data['img'][0].data.dtype == torch.uint8:
            data['img'][0] = data['img'][0].float()
        data = collate([data], samples_per_gpu=1)
        if device == 'cuda':
            data = scatter(data, [device])[0]
    return data


def inference_classifier(model, data):
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_scores = np.max(scores, axis=1)
        pred_labels = np.argmax(scores, axis=1)

    results = []
    for pred_score, pred_label in zip(pred_scores, pred_labels):
        pred_label = int(pred_label)
        pred_score = float(pred_score)
        result = {'pred_label': pred_label,
                  'pred_score': pred_score,
                  'pred_class': model.CLASSES[pred_label]}
        results.append(result)
    return results


def draw_cls_result(img_or_path, result, model, **kwargs):
    img = mmcv.imread(img_or_path)
    img_drawn = model.show_result(img, result, **kwargs)
    return img_drawn
