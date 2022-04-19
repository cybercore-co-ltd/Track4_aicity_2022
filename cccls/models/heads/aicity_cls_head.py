import torch
import torch.nn.functional as F

from mmcls.models import HEADS, LinearClsHead


@HEADS.register_module()
class AICityLinearClsHead(LinearClsHead):
    """Support TTA"""

    def aug_test(self, cls_scores, softmax=True, post_process=True):
        """Inference with augmentation.

        Args:
            cls_scores (List[tuple[Tensor]]):
                The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        aug_pred = []
        for cls_score in cls_scores:
            pred = self.simple_test(
                cls_score, softmax=True, post_process=False)
            aug_pred.append(pred)
        aug_pred = torch.stack(aug_pred).mean(dim=0)

        # if softmax:
        #     aug_pred = F.softmax(aug_pred, dim=1)
        # else:
        #     aug_pred = cls_score

        if post_process:
            return self.post_process(aug_pred)
        else:
            return aug_pred
