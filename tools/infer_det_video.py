import os
import time
import mmcv
import cvut
import torch
import argparse
from ccdet.apis import (build_det_data_pipeline, init_detector, get_det_data,
                        inference_detector, filter_informative_coco_classes)


# ------------------------------------------------------------------------------
#  ArgumentParser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--cfg", type=str, help="Config file")

parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint file")

parser.add_argument("--video", type=str, help="Video file")

parser.add_argument("--fp16", action='store_true', default=False,
                    help="Enable FP16")

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--det-thr", type=float, default=0.3,
                    help="Detection threshold")

parser.add_argument("--out-dir", type=str, default="cache/det/")

parser.add_argument("--save-pred", action='store_true', default=False,
                    help="Save prediction")

parser.add_argument("--coco-filter", action='store_true', default=False,
                    help="Filter informative COCO classes")

args = parser.parse_args()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
def main():
    # 6 classes when using model trained on COCO
    INFORMATIVE_COCO_CLASSES = ["book", "cup", "bottle", "hair drier",
                                "toothbrush", "remote"]
    cap, (_, _), num_frames, fps = cvut.get_video(args.video)

    # build data_pipeline and model
    cfg = mmcv.Config.fromfile(args.cfg)
    data_pipeline = build_det_data_pipeline(cfg.test_pipeline)
    model = init_detector(args.cfg, args.ckpt,
                          device=args.device, fp16=args.fp16)

    # inference
    preds = []
    start_time = time.perf_counter()
    for frame_idx in range(num_frames):

        # read frame
        status, frame = cap.read()
        if not status:
            break
        image = frame

        # preprocessing
        data = get_det_data(image, data_pipeline, device=args.device)

        # model inference
        tic = time.perf_counter()
        results = inference_detector(model, data)
        if len(results) == 2:  # bbox, mask
            results = results[0]  # bbox

        if (len(results) == 80) and args.coco_filter:
            results = filter_informative_coco_classes(results,
                                                      INFORMATIVE_COCO_CLASSES)
        runtime = time.perf_counter() - tic

        print("\n[{}/{}] runtime: {} [ms]".format(frame_idx +
                                                  1, num_frames, int(1e3*runtime)))

        # collect predictions
        if args.save_pred:
            preds.append((frame_idx, results))

    # total runtime
    runtime = time.perf_counter() - start_time
    print("\nTotal runtime: {:.6f} [s]".format(runtime))

    # save predictions
    if args.save_pred:
        basename = cvut.basename(args.video, wo_fmt=True)
        pkl_file = os.path.join(args.out_dir, "{}.pkl".format(basename))
        mmcv.dump(preds, pkl_file)
        print("Result is saved at {}".format(pkl_file))

if __name__ == "__main__":
    with torch.no_grad():
        main()
