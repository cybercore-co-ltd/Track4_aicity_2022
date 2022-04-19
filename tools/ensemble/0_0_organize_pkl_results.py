from glob import glob
import os
import os.path as osp
import shutil

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="./data/full_video_pkl")
    parser.add_argument('--save_dirs', default="./cache/process_results")
    args = parser.parse_args()
    return args


args = parse_args()

all_models = os.listdir(args.input_dir)

all_results = [glob(osp.join(args.input_dir, all_models[idx] + '/*.pkl'))
               for idx in range(len(all_models))]

all_results = [sorted(x) for x in all_results]

for idx, _results in enumerate(list(zip(*all_results))):
    _save_dir = osp.join(args.save_dirs, osp.splitext(
        osp.basename(_results[0]))[0])
    os.makedirs(_save_dir, exist_ok=True)
    for _idx, _re in enumerate(_results):
        shutil.copy(_re, osp.join(_save_dir, all_models[_idx])+'.pkl')
    print('pkls are save at: ', _save_dir)
