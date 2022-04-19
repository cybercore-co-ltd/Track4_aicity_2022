from mmdet.datasets import PIPELINES
import numpy as np

@PIPELINES.register_module()
class ClassAgnostic:
    def __init__(self, repeat_lable=False, lable_value=0, load_gt_ratio=False):
        self.repeat_lable = repeat_lable
        self.lable_value = lable_value
        self.load_gt_ratio = load_gt_ratio
    def __call__(self, results):
        if self.repeat_lable:
            results['gt_labels'] = np.repeat(np.array([73, 41, 39, 78, 79, 65]), len(results['gt_bboxes']), 0)
            results['gt_bboxes'] = np.repeat(results['gt_bboxes'], 6, 0)
            if 'gt_masks' in results:
                results['gt_masks'].masks = np.repeat(results['gt_masks'], 6, 0)
        elif not self.load_gt_ratio:
            results['gt_labels'][:] = self.lable_value
        
        return results
