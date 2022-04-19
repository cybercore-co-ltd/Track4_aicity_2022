import argparse
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from ccdet import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--pkl', help='result pkl')
    parser.add_argument('--data_type', default='val')
    parser.add_argument('--eval', type=str, default='bbox')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    if args.data_type == 'test':
        dataset = build_dataset(cfg.data.test)
    elif args.data_type == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.data_type == 'train':
        dataset = build_dataset(cfg.data.train)

    outputs = mmcv.load(args.pkl)
    eval_kwargs = dict(metric=args.eval)
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)

if __name__ == '__main__':
    main()
