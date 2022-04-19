import mmcv
from glob import glob
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_pkls_dir')
    parser.add_argument('--original_pkls_dir')
    args = parser.parse_args()
    return args

args = parse_args()

all_ensembles = sorted(glob(os.path.join(args.ensemble_pkls_dir, '*.pkl')))
all_originals = sorted(glob(os.path.join(args.original_pkls_dir, '*.pkl')))
assert len(all_ensembles) == len(all_originals)
for file_en, file_ori in list(zip(all_ensembles, all_originals)):
    assert os.path.basename(file_en) == os.path.basename(file_ori)
    save_data = []
    en_res = mmcv.load(file_en)
    ori_res = mmcv.load(file_ori)
    for x, y in zip(en_res, ori_res):
        save_data.append((y[0], x))
    mmcv.dump(save_data, file_en)
    print('pkl is dumped at: ', file_en)
    