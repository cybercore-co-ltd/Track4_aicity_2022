import numpy as np
from mmdet.datasets import CocoDataset, DATASETS
from ccdet.apis import filter_informative_coco_classes


@DATASETS.register_module()
class CocoAgnosticTest(CocoDataset):
    def evaluate(self,
                 results=None,
                 *args,
                 **kwargs):
        # in case of (bbox, mask), only take bbox
        if len(results[0]) == 2:
            results = [item[0] for item in results]
        if len(results[0]) == 80:
            INFORMATIVE_COCO_CLASSES = ["book", "cup", "bottle", "hair drier",
                                        "toothbrush", "remote"]
            results = [filter_informative_coco_classes(result,
                                                       INFORMATIVE_COCO_CLASSES)
                       for result in results]
        # merge multiple classes into one
        results = [[np.concatenate(result), ] for result in results]
        return super().evaluate(results=results, *args, **kwargs)
