import torch
import torch.nn as nn
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import StandardRoIHead
import torch.nn.functional as F


@HEADS.register_module()
class LWOF_StandardRoIHead(StandardRoIHead):
    def __init__(self, loss_bbox_similar=None, loss_cls_similar=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_bbox_similar is not None:
            self.loss_bbox_similar = build_loss(loss_bbox_similar)

        if loss_cls_similar is not None:
            if loss_cls_similar['type'] == "KLLoss":
                self.kl_loss = True
                self.loss_cls_similar = nn.KLDivLoss(reduction="batchmean")
                self.loss_cls_weight = loss_cls_similar['loss_weight']
            else:
                self.loss_cls_similar = build_loss(loss_cls_similar)
        self.loss_source_cls = nn.CrossEntropyLoss()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      source_target_rel=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        # bbox head forward and loss

        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas,
                                                    source_target_rel=source_target_rel)
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            bbox_results.update(mask_results['loss_mask'])

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, source_target_rel=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        if source_target_rel is None:
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)
        else:
            # Co-tuning
            source_gt_labels = []
            for gt_label in bbox_targets[0]:
                source_gt_labels.append(source_target_rel[gt_label])

            source_gt_labels = torch.cat(source_gt_labels, 0).\
                reshape(-1, self.bbox_head.num_classes+1)

            loss_bbox = dict()
            loss_bbox['loss_cls'] = self.loss_source_cls(
                bbox_results['cls_score'], source_gt_labels)

        bbox_results.update(loss_bbox=loss_bbox)
        # add target labels information to bbox_results
        if source_target_rel is None:
            bbox_results.update(target_labels=bbox_targets[0])
        return bbox_results

    def loss_similar(self, losses, st_bbox_preds, te_bbox_preds, st_cls_score, te_cls_score):
        loss_bbox_sim = self.loss_bbox_similar(te_bbox_preds, st_bbox_preds)

        if self.kl_loss:
            input = F.log_softmax(te_cls_score, dim=1)
            target = F.softmax(st_cls_score, dim=1)
            loss_cls_sim = self.loss_cls_weight * \
                self.loss_cls_similar(input, target)
        else:
            raise NotImplementedError

        losses.update(loss_bbox_sim=loss_bbox_sim)
        losses.update(loss_cls_sim=loss_cls_sim)
        return losses
