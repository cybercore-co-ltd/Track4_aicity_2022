from mmdet.models.detectors import FasterRCNN
from mmdet.models import (DETECTORS,
                          build_backbone, build_neck, build_head)
import torch
import torch.nn as nn
from mmcv.runner import _load_checkpoint
from copy import deepcopy


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def category_relationship(pretrain_prob, target_labels):
    """
    The direct approach of learning category relationship.
    :param pretrain_prob: shape [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param target_labels:  shape [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :return: shape [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """
    N_t = torch.max(target_labels) + 1
    conditional = []

    for i in range(N_t):
        this_class = pretrain_prob[target_labels == i]
        average = torch.mean(this_class, axis=0, keepdims=True)
        conditional.append(average)
    return torch.cat(conditional)


@DETECTORS.register_module()
class LWOF_FasterRCNN(FasterRCNN):
    """Implementation of \"Learning Without Forgeting\" with FasterRCNN as a base model"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 # old model
                 teacher_backbone,
                 teacher_rpn_head,
                 teacher_roi_head,
                 teacher_neck=None,
                 teacher_pretrained=None,
                 novel_roi_head=None,
                 # new model
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 novel_roi_head_pretrained=None):

        super(LWOF_FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.loss_weight = 2.
        # build teacher model
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = build_neck(teacher_neck)
        if teacher_rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = teacher_rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.teacher_model.rpn_head = build_head(rpn_head_)

        if teacher_roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            teacher_roi_head.update(train_cfg=rcnn_train_cfg)
            teacher_roi_head.update(test_cfg=test_cfg.rcnn)
            teacher_roi_head.pretrained = pretrained
            self.teacher_model.roi_head = build_head(teacher_roi_head)

        if novel_roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            novel_roi_head.update(train_cfg=rcnn_train_cfg)
            novel_roi_head.update(test_cfg=test_cfg.rcnn)
            novel_roi_head.pretrained = pretrained
            self.novel_roi_head = build_head(novel_roi_head)

        # use the same pretrained model for teacher and student
        self.teacher_pretrained = teacher_pretrained
        self.novel_roi_head_pretrained = novel_roi_head_pretrained

    def init_weights(self):
        super().init_weights()
        self.init_teacher_weights(self.teacher_pretrained)
        self.init_novel_roi_head(self.novel_roi_head_pretrained)

    def init_teacher_weights(self, teacher_pretrained):
        # load ckpt
        ckpt = _load_checkpoint(teacher_pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        # build new ckpt
        self_state_dict = self.state_dict()
        new_ckpt = deepcopy(self_state_dict)
        for key in self_state_dict:
            if 'teacher_' in key:
                src_key = key.replace('teacher_model.', '')
                new_ckpt[key] = ckpt[src_key]
        # load state_dict
        self.load_state_dict(new_ckpt, strict=False)
        print(f"Init teacher weights from {teacher_pretrained}")

    def init_novel_roi_head(self, novel_roi_head_pretrained):
        # load ckpt
        ckpt = _load_checkpoint(novel_roi_head_pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        # build new ckpt
        self_state_dict = self.state_dict()
        new_ckpt = deepcopy(self_state_dict)
        new_ckpt2 = deepcopy(self_state_dict)
        for key, val in new_ckpt2.items():
            if not 'novel_roi_head' in key:
                new_ckpt.pop(key)
        for key in self_state_dict:
            if 'novel_roi_head' in key:
                src_key = key.replace('novel_', '')
                new_ckpt[key] = ckpt[src_key]
        # load state_dict
        self.load_state_dict(new_ckpt, strict=False)
        print(f"Init novel roi head weights from {novel_roi_head_pretrained}")

    def teacher_eval(self):
        self.teacher_model.backbone.eval()
        self.teacher_model.neck.eval()
        self.teacher_model.rpn_head.eval()
        self.teacher_model.roi_head.eval()

    @property
    def with_teacher_neck(self):
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

    def update_loss_weight(self, loss_weight):
        # print(f'loss_weight changed from {self.loss_weight} to {loss_weight}')
        self.loss_weight = loss_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        # forward to teacher model
        with torch.no_grad():
            # self.teacher_model.backbone.stages[2].blocks[17].attn.w_msa.proj.weight[0].detach().cpu().numpy()

            self.teacher_eval()
            x_teacher = self.extract_teacher_feat(img)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.with_rpn:
                te_rpn_losses, te_proposal_list = self.teacher_model.rpn_head.forward_train(
                    x_teacher,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)

            else:
                te_proposal_list = proposals

            # forward x_teacher to teacher roi_head
            te_bbox_results = self.teacher_model.roi_head.forward_train(x_teacher,
                                                                        img_metas, te_proposal_list,
                                                                        gt_bboxes, gt_labels,
                                                                        gt_bboxes_ignore, gt_masks,
                                                                        **kwargs)
        losses = dict()
        # forward to student model
        x_student = self.extract_feat(img)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, st_proposal_list = self.rpn_head.forward_train(
                x_student,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            st_proposal_list = proposals

        st_bbox_results = self.roi_head.forward_train(x_student, img_metas, st_proposal_list,
                                                      gt_bboxes, gt_labels,
                                                      gt_bboxes_ignore, gt_masks,
                                                      **kwargs)

        # distillation loss
        te_bbox_preds = te_bbox_results['bbox_pred']
        te_cls_score = te_bbox_results['cls_score']

        st_bbox_preds = st_bbox_results['bbox_pred']
        st_cls_score = st_bbox_results['cls_score']

        # major loss
        losses = self.roi_head.loss_similar(losses, te_bbox_preds,
                                            st_bbox_preds,
                                            te_cls_score,
                                            st_cls_score)

        # student loss
        novel_bbox_results = self.novel_roi_head.forward_train(x_student, img_metas, st_proposal_list,
                                                               gt_bboxes, gt_labels,
                                                               gt_bboxes_ignore, gt_masks,
                                                               **kwargs)
        losses.update(loss_roi_mask=self.loss_weight *
                      st_bbox_results['loss_mask'])
        losses.update(novel_bbox_results)
        losses.update(
            st_weight=1.0*torch.tensor(self.loss_weight).to(st_bbox_results['loss_mask'].device))

        te_rpn_losses_updated = dict()
        for key, value in te_rpn_losses.items():
            te_rpn_losses_updated['te_'+key] = value

        losses.update(te_rpn_losses_updated)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if self.test_cfg.use_novel_roi_head:
            roi_head = self.novel_roi_head
        else:
            roi_head = self.roi_head
        return roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        if self.test_cfg.use_novel_roi_head:
            roi_head = self.novel_roi_head
        else:
            roi_head = self.roi_head
        return roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
