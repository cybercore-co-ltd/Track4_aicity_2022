import mmcv
from sklearn.utils import assert_all_finite
from ccdet.ops import weighted_boxes_fusion
import numpy as np
import copy


def wbf(origin_pkl_datas, dataset, weights=[1, 1],
        iou_thr=0.7, video_H=None, video_W=None, video_numclasses=1):
    use_dataset = not (video_H is not None and video_W is not None)
    pkl_datas = []
    for x in origin_pkl_datas:
        pkl_datas.append(copy.deepcopy(x))
    # mmdet pkl style
    model_num = len(pkl_datas)
    bboxes_lists = [[] for _ in range(model_num)]
    labels_lists = [[] for _ in range(model_num)]
    scores_lists = [[] for _ in range(model_num)]
    for i_pkl, model_result in enumerate(pkl_datas):
        # Scale x1, y1, x2, y2 to [0, 1]
        for idx in range(len(model_result)):
            if not use_dataset:
                H = video_H
                W = video_W
            else:
                img_info = dataset.data_infos[idx]
                H, W = img_info['height'], img_info['width']

            for i in range(len(model_result[idx])):
                model_result[idx][i][:, 0] /= W
                model_result[idx][i][:, 2] /= W
                model_result[idx][i][:, 1] /= H
                model_result[idx][i][:, 3] /= H

            valid_id = [True]*len(np.concatenate(model_result[idx])[:, 4])

            concat_bboxes = np.concatenate(model_result[idx])[:, :4]
            bboxes_lists[i_pkl].append(
                [concat_bboxes[i] for i, _v in enumerate(valid_id) if _v])

            concat_scores = np.concatenate(model_result[idx])[:, 4]
            scores_lists[i_pkl].append(
                [concat_scores[i] for i, _v in enumerate(valid_id) if _v])

            # make label list
            _img_label = []
            for i in range(len(model_result[idx])):
                if model_result[idx][i].any():
                    _img_label += [i]*len(model_result[idx][i])
            labels_lists[i_pkl].append([_img_label[i]
                                        for i, _v in enumerate(valid_id) if _v])

    bboxes_fusion_lists = []
    labels_fusion_lists = []
    scores_fusion_lists = []
    for idx in range(len(model_result)):
        img_boxes_lists = [bboxes_lists[i][idx] for i in range(model_num)]
        img_labels_lists = [labels_lists[i][idx] for i in range(model_num)]
        img_scores_lists = [scores_lists[i][idx] for i in range(model_num)]

        bboxes_fusion, scores_fusion, labels_fusion = \
            weighted_boxes_fusion(
                img_boxes_lists, img_scores_lists,
                img_labels_lists, weights=weights,
                iou_thr=iou_thr, skip_box_thr=0.01,
                conf_type='max')
        bboxes_fusion_lists.append(bboxes_fusion)
        scores_fusion_lists.append(scores_fusion)
        labels_fusion_lists.append(labels_fusion)

    # Rescale to origin shape
    for idx in range(len(model_result)):
        if not use_dataset:
            H = video_H
            W = video_W
        else:
            img_info = dataset.data_infos[idx]
            H, W = img_info['height'], img_info['width']

        bboxes_fusion_lists[idx][:, 0] *= W
        bboxes_fusion_lists[idx][:, 2] *= W
        bboxes_fusion_lists[idx][:, 1] *= H
        bboxes_fusion_lists[idx][:, 3] *= H

    # Regenerate the original pkl file
    bboxes_scores_lists = [np.concatenate((bboxes_fusion_lists[idx],
                                           scores_fusion_lists[idx].reshape(-1, 1)),
                                          axis=1) for idx in range(len(model_result))]

    # Build output
    if not use_dataset:
        assert video_numclasses is not None
        num_classes = video_numclasses
    else:
        num_classes = len(dataset.CLASSES)
    outputs = [[] for __ in range(len(model_result))]

    for idx in range(len(model_result)):
        for i in range(num_classes):
            if (labels_fusion_lists[idx] == i).any():
                outputs[idx].append(np.concatenate(
                    bboxes_scores_lists[idx][labels_fusion_lists[idx] == i],
                    axis=0).reshape(-1, 5))
            else:
                outputs[idx].append(np.zeros([0, 5]))
    return outputs


def multi_process_wbf(global_list, pkl_datas, dataset, args):
    # mmdet pkl style
    weights = args[0]
    iou_thr = args[1]
    outputs = wbf(pkl_datas, dataset, weights=weights, iou_thr=iou_thr)
    mAP = dataset.evaluate(outputs, classwise=True)
    global_list.append({'mAP': mAP, 'weights': weights,
                        'iou': iou_thr})
