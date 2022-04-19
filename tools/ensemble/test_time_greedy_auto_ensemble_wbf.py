import mmcv
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from ccdet.datasets import *
import glob
from ccdet.ops.wbf_function import wbf
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, help="containing data")
parser.add_argument("pkl_dir", type=str, help="dir to .pkl files")
parser.add_argument("save_test_time_ensemble_name",
                    type=str,
                    help="save ensemble file")
parser.add_argument("log_tree_file",
                    type=str,
                    help="greedy search log tree (.pkl)")

parser.add_argument('--evaluate', action='store_true', help='evaluate?')
parser.add_argument('--dataset-type', default='test')
parser.add_argument('--video_dir', default=None, type=str)
parser.add_argument('--video_numclasses', default=1, type=int)

args = parser.parse_args()

config_file = args.config_file
pkl_dir = args.pkl_dir
save_test_time_ensemble_name = args.save_test_time_ensemble_name
log_tree_file = args.log_tree_file

if args.video_dir is not None:
    vcap = cv2.VideoCapture(args.video_dir)  # 0=camera
    video_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
    video_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
else:
    # Build dataset
    print('build from config')
    cfg = Config.fromfile(config_file)
    if args.dataset_type == 'test':
        dataset = build_dataset(cfg.data.test)
    elif args.dataset_type == 'val':
        dataset = build_dataset(cfg.data.val)
    else:
        dataset = build_dataset(cfg.data.train)
    print('len dataset: ', len(dataset))

# reweight classification score


def reweight_cls_score(pkldata, weight):
    for img_data in pkldata:
        for box_score in img_data:
            if len(box_score) != 0:
                box_score[:, 4] *= weight


pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))

log_tree = mmcv.load(log_tree_file)

print(log_tree)
for layer_idx, layer in enumerate(tqdm(log_tree)):
    layer_pkl_datas = []

    if layer_idx == 0:
        pkl_name_test_time_list = layer.pop(-1)
        pkl_dir_test_time_list = [os.path.join(pkl_dir, _pkl_file)
                                  for _pkl_file in pkl_name_test_time_list]
        pkl_datas = [0] * len(pkl_dir_test_time_list)
        for file_idx, _test_file_name in enumerate(pkl_files):
            pkl_datas[file_idx] = mmcv.load(_test_file_name)

    refactor = layer[-1]
    nodes = []
    for _ii in range(len(layer) - 1):
        nodes.append(layer[_ii])

    for pair in nodes:
        indexes = pair['indexes']
        iou = pair['iou']
        weight = pair['weight']
        if len(indexes) == 1:
            results = pkl_datas[indexes[0]]
        else:
            # reweight
            for _idx in range(len(indexes)):
                reweight_cls_score(pkl_datas[indexes[_idx]],
                                   refactor[indexes[_idx]])
            if args.video_dir is not None:
                results = wbf([pkl_datas[i] for i in indexes],
                              None, weights=weight, iou_thr=iou,
                              video_H=video_height, video_W=video_width, 
                              video_numclasses=args.video_numclasses)
            else:
                results = wbf([pkl_datas[i] for i in indexes],
                              dataset, weights=weight, iou_thr=iou)

        layer_pkl_datas.append(results)
    pkl_datas = layer_pkl_datas.copy()

os.makedirs(os.path.dirname(save_test_time_ensemble_name), exist_ok=True)
print('test time ensemble is saving at: ', save_test_time_ensemble_name)
mmcv.dump(pkl_datas[0], save_test_time_ensemble_name)

try:
    if args.evaluate:
        print('evaluate after ensemble......................')
        for x in pkl_datas:
            dataset.evaluate(x)
except:
    pass
