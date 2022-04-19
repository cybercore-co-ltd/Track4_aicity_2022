import mmcv
from glob import glob
import os
from ccdet.apis import filter_informative_coco_classes
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pkl_dir')
    parser.add_argument('--input_pkl_dir')
    args = parser.parse_args()
    return args

args = parse_args()

save_pkl_dir = args.save_pkl_dir
input_pkl_dir = args.input_pkl_dir

# save_pkl_dir = './cache/save_val_processed_pkl_dir'
# input_pkl_dir = "data/val_pkl_results/save_val_processed_pkl_dir/*.pkl"

os.makedirs(save_pkl_dir, exist_ok=True)
pkls = glob(os.path.join(input_pkl_dir, '*.pkl'))
INFORMATIVE_COCO_CLASSES = ["book", "cup", "bottle", "hair drier",
                            "toothbrush", "remote"]
for file in tqdm(pkls):
    
    data = mmcv.load(file)
    if isinstance(data[0][0], int):
        data = [x[1] for x in data]
    # in case of (bbox, mask), only take bbox
    if len(data[0]) == 2:
        data = [item[0] for item in data]
    if len(data[0]) == 80:
        data = [filter_informative_coco_classes(result,
                                        INFORMATIVE_COCO_CLASSES)
                    for result in data]
    # merge multiple classes into one
    data = [[np.concatenate(result), ] for result in data]
    save_file = os.path.join(save_pkl_dir, os.path.basename(file))
    mmcv.dump(data, save_file)
    print('pkl file is dumped at: ', save_file)
    