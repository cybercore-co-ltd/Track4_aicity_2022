import cv2
import numpy as np
import random
import os
import json
# Util methods for augmentation

path = os.path.abspath("../../data/random_bg")
if not os.path.exists(path):
    os.makedirs(path)

save_dict = {'images': [], 'annotations': []}
save_dict["categories"] = [
    {"supercategory": "person", "id": 0, "name": "person"}
]

img_id = 0
annotation_id = 0
for i in range(320):
    value = random.randint(120, 220)
    img = np.zeros([1080, 1920, 3], dtype=np.uint8)
    noise = np.random.normal(255./2, 255./10, img.shape)
    img[:] = value
    H, W = img.shape[:2]
    img[H-300:, ...] = noise[H-300:, ...]

    filename = str(i).zfill(6) + '.jpg'
    filepath = os.path.join('../../data/random_bg', filename)

    cv2.imwrite(filepath, img)

    gt_bboxes = [0, 0, 0, 0]
    gt_labels = 0
    gt_masks = None
    img_name = str(i).zfill(6) + '.jpg'
    new_image = {
        "license": 4,
        "file_name": img_name,
        "coco_url": "",
        "height": img.shape[0],
        "width": img.shape[1],
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "",
        "id": img_id,
    }

    new_ann = {
        "segmentation": [],
        "area": 1080*1920,
        "iscrowd": 0,
        "image_id": img_id,
        "bbox": [0, 0, 100, 100],
        # "bbox": bbox,
        "category_id": 0,
        "id": annotation_id
    }

    save_dict['annotations'].append(new_ann)
    save_dict['images'].append(new_image)
    img_id += 1
    annotation_id += 1

ann_path = os.path.abspath("../../data/annotations/")
if not os.path.exists(ann_path):
    os.makedirs(ann_path)

with open(os.path.join(ann_path, 'random_bg.json'), 'w') as fp:
    json.dump(save_dict, fp)
