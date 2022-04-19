import mmcv
import cvut
import torch
import numpy as np
import cv2
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import collate, scatter

from mmdet.core import multiclass_nms
from mmdet.datasets import CocoDataset
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.runner import load_checkpoint, wrap_fp16_model


def build_data_pipeline(test_pipeline):
    test_pipeline[0] = LoadImage()
    data_pipeline = Compose(test_pipeline)
    return data_pipeline


def inference_tracker(tracker, result):
    det_bboxes = np.concatenate(result)
    if det_bboxes.shape[-1] == 6:
        # det_bboxes contains mask ratio
        track_ids, det_bboxes, mask_ratios = tracker(det_bboxes)
        return track_ids, det_bboxes, mask_ratios
    else:
        track_ids, det_bboxes = tracker(det_bboxes)
        return track_ids, det_bboxes


def draw_track_result(track_ids, det_bboxes, det_labels,
                      img_or_path, classnames, mask_ratios=None, thickness=4,
                      font_size=1.0, font_thickness=2,):
    num_objs = len(det_bboxes)
    if det_bboxes.shape[1] == 5:
        det_bboxes = det_bboxes[:, :4]
    # draw img
    ori_img = mmcv.imread(img_or_path)
    img = cvut.draw_track(ori_img, det_bboxes, track_ids, det_labels,
                          classnames=classnames, thickness=thickness,
                          font_size=font_size, font_thickness=font_thickness)
    if mask_ratios is not None:
        img = draw_mask_ratio_result(img, mask_ratios, det_bboxes)
    return img, ori_img, num_objs


def init_detector(config, checkpoint=None, device='cuda', fp16=False):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    # Parse config
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    # Build model
    config.model.pretrained = None
    model = build_detector(config.model)

    # Load checkpoint
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if ('meta' in checkpoint) and ('CLASSES' in checkpoint['meta']):
            model.CLASSES = checkpoint['meta']['CLASSES']
    if "fuse_convbn" in config and not config.fuse_convbn:
        pass
    else:
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

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def build_det_data_pipeline(test_pipeline):
    test_pipeline[0] = LoadImage()
    data_pipeline = Compose(test_pipeline)
    return data_pipeline


def get_det_data(img, data_pipeline, device='cuda'):
    """Support both cpu and cuda device"""
    data = dict(img=img)
    data = data_pipeline(data)
    if data['img'][0].data.dtype == torch.uint8:
        data['img'][0] = data['img'][0].float()
    data = collate([data], samples_per_gpu=1)
    if isinstance(device, str) and (device == 'cpu'):
        data['img_metas'] = data['img_metas'][0].data
    else:
        data = scatter(data, [device])[0]
    return data


def inference_detector(model, data, **kwargs):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data, **kwargs)
        # b/c bs=1, so we take the first (only) sample
        result = result[0]
    return result


def draw_det_result(result, img_or_path, classnames, det_thr, ratios=None,
                    color=(255, 255, 255)):
    if len(result) == 2:  # bbox, mask
        result = result[0]
    bboxes = np.concatenate(result)
    scores = bboxes[:, 4]
    bboxes = bboxes[:, :4]
    labels = np.concatenate([
        idx * np.ones([len(item)]) for idx, item in enumerate(result)])
    # filter high-confident samples
    inds = scores >= det_thr
    bboxes = bboxes[inds].astype(int)
    labels = labels[inds].astype(int)
    scores = scores[inds]
    num_objs = len(bboxes)
    # draw img
    ori_img = mmcv.imread(img_or_path)
    img = cvut.draw_bboxes(ori_img, bboxes, labels, scores, classnames,
                           color=color, thickness=2, font_size=1.0)
    if ratios is not None:
        img = draw_mask_ratio_result(img, ratios, bboxes)
    return img, ori_img, bboxes, num_objs


def draw_mask_ratio_result(img, ratios, bboxes):
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_size = 1
    _font_thickness = 2
    _color = (255, 255, 255)
    for (bbox, ratio) in zip(bboxes, ratios):
        x1, y1, _, _ = [int(ele) for ele in bbox]
        ratio = np.round(ratio, 2)
        img = cv2.putText(img, str(ratio), (x1+5, y1+30), _font,
                          _font_size, _color, thickness=_font_thickness)
    return img


def draw_mask_result(bboxes, masks, img_or_path, det_thr):
    bboxes = np.concatenate(bboxes)
    scores = bboxes[:, 4]
    masks = np.concatenate(masks)
    # filter high-confident samples
    inds = scores >= det_thr
    masks = masks[inds]
    # draw mask
    ori_img = mmcv.imread(img_or_path)
    img = cvut.draw_inst_masks(ori_img, masks)
    return img, ori_img, masks


def filter_informative_coco_classes(results, informative_cls_names,
                                    shape_thr=(100, 100), masks=None, with_nms=True):
    CLASSES = CocoDataset.CLASSES
    cls_inds = [CLASSES.index(info_cls) for info_cls in informative_cls_names]
    results = [results[idx] for idx in cls_inds]
    results = np.concatenate(results)

    if masks is not None:
        tmp = [np.stack(masks[idx]) for idx in cls_inds if len(masks[idx]) > 0]
        if len(tmp) > 0:
            masks = np.concatenate(tmp)
        else:
            masks = np.array([])

    # shape_thr (width, height) of bboxes
    if shape_thr is not None:
        bboxes = results[:, :4]
        ws = bboxes[:, 2] - bboxes[:, 0]
        hs = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (ws >= shape_thr[0]) * (hs >= shape_thr[1])
        results = results[valid_inds]
        if masks is not None:
            masks = masks[valid_inds]

    # sort by score
    scores = results[:, -1]
    inds = np.argsort(scores)[::-1]
    results = results[inds]
    if masks is not None:
        masks = masks[inds]
    # nms
    if with_nms:
        bboxes = torch.from_numpy(results[:, :4])
        scores = torch.cat([torch.from_numpy(results[:, 4:5]),
                            torch.zeros([len(results), 1])], dim=1)
        dets, _, return_inds = multiclass_nms(bboxes, scores, score_thr=0.01, max_num=100,
                                              nms_cfg=dict(type='nms', iou_threshold=0.6), return_inds=True)
        results = dets.numpy()
        if masks is not None:
            masks = masks[return_inds]
    if masks is not None and len(masks.shape) == 2:
        masks = np.expand_dims(masks, 0)
        assert len(results) == len(
            masks), f'len bboxs {len(results)} and masks {len(masks)} are not equal'
        return [results], masks
    else:
        return [results]
