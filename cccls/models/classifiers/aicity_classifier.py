import torch
from mmcls.models import CLASSIFIERS, ImageClassifier, MultiLabelClsHead


@CLASSIFIERS.register_module()
class AICityImageClassifier(ImageClassifier):
    """Support TTA"""

    def forward_test(self, imgs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            return self.aug_test(imgs, **kwargs)

    def aug_test(self, imgs, img_metas=None, **kwargs):
        """Test function with test time augmentation."""
        x = [self.extract_feat(img) for img in imgs]

        try:
            if isinstance(self.head, MultiLabelClsHead):
                assert 'softmax' not in kwargs, (
                    'Please use `sigmoid` instead of `softmax` '
                    'in multi-label tasks.')
            res = self.head.aug_test(x, **kwargs)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        return res
