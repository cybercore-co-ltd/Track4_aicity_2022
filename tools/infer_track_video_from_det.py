import os
import time
import mmcv
import cvut
import torch
import argparse
import numpy as np

from ccdet.models import BYTETracker
from ccdet.apis import (inference_tracker, draw_track_result,
                        filter_informative_coco_classes)


# ------------------------------------------------------------------------------
#  ArgumentParser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--pkl", type=str, help="Pkl file")

parser.add_argument("--video", type=str, help="Video file")

parser.add_argument("--fp16", action='store_true', default=False,
                    help="Enable FP16")

parser.add_argument("--reverse", action='store_true', default=False,
                    help="Reverse video")

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--det-thr", type=float, default=0.3,
                    help="Detection threshold")

parser.add_argument("--match-thr", type=float, default=0.7,
                    help="Match threshold")

parser.add_argument("--track-buffer", type=int, default=30,
                    help="Track buffer")

parser.add_argument("--start-frame", type=int, default=0,
                    help="Start frame for visualization")

parser.add_argument("--max-frames", type=int, default=-1,
                    help="Number of maximum frames for visualization")

parser.add_argument("--out-dir", type=str, default="cache/track",
                    help="Output directory to save visualization results")

parser.add_argument("--save-img", action='store_true', default=False,
                    help="Save image")

parser.add_argument("--save-pred", action='store_true', default=False,
                    help="Save prediction")

parser.add_argument("--save-video", action='store_true', default=False,
                    help="Save video")

args = parser.parse_args()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
def main():
    assert not (args.reverse and args.save_video)

    # FRCNN-SwinS
    INFORMATIVE_COCO_CLASSES = ["book", "cup", "bottle", "hair drier",
                                "toothbrush", "remote"]

    # input video
    cap, (width, height), num_frames, fps = cvut.get_video(args.video)
    print(
        f"[Video] FPS: {fps}; Resolution: {(width, height)}; num_frames: {num_frames}")
    if num_frames == 0:
        num_frames = int(1e8)

    # output video
    if args.save_video:
        out_video_file = os.path.join(args.out_dir,
                                      cvut.basename(args.video, wo_fmt=True))
        out_video_file = out_video_file + '.mp4'
        out = cvut.create_video(out_video_file, (width, height), fps=fps)

    # build tracker
    tracker = BYTETracker(track_thresh=args.det_thr,
                          match_thresh=args.match_thr,
                          track_buffer=args.track_buffer,
                          fuse_det_score=False,
                          frame_rate=fps)

    # load detection outputs from pkl file
    outputs = mmcv.load(args.pkl)

    # reverse
    if args.reverse:
        outputs = outputs[::-1]

    # get start and stop frame
    start_frame = args.start_frame if args.start_frame != -1 else 0
    stop_frame = (start_frame + args.max_frames if args.max_frames != -1
                  else num_frames)

    # get dataset
    CLASSES = ('object', )

    # inference
    preds = []
    start_time = time.perf_counter()
    for frame_idx in range(num_frames):

        # start condition
        if frame_idx < start_frame:
            status, image = cap.read()
            print(f"Skip frame {frame_idx} < {start_frame}")
            continue

        # stop condition
        if frame_idx >= stop_frame-1:
            break

        # read frame
        if args.save_video:
            status, image = cap.read()
            if not status:
                break

        # get detection
        tic = time.perf_counter()
        if frame_idx >= len(outputs):
            break
        _frame_idx, results = outputs[frame_idx]
        if not args.reverse:
            assert _frame_idx == frame_idx

        if len(results) == 2:  # bbox, mask
            results = results[0]

        if len(results) == 80:
            results = filter_informative_coco_classes(results,
                                                      INFORMATIVE_COCO_CLASSES)

        # track
        track_ids, det_bboxes = inference_tracker(tracker, results)
        if len(results) == 80:
            det_labels = np.zeros([len(det_bboxes)], dtype=int)
        else:
            det_labels = None
        runtime = time.perf_counter() - tic

        # draw result
        if args.save_video:
            img, _, num_objs = draw_track_result(track_ids, det_bboxes, det_labels,
                                                 image, CLASSES)
            print("\n[{}/{}] runtime: {} [ms] | num_objs: {}".format(
                frame_idx+1, num_frames, int(1e3*runtime), num_objs))
            out.write(img)
        else:
            print("\n[{}/{}] runtime: {} [ms]".format(
                frame_idx+1, num_frames, int(1e3*runtime)))

        # write result image
        if args.save_img:
            out_file = os.path.join(
                args.out_dir, f"frame_{str(frame_idx).zfill(8)}.jpg")
            cvut.imwrite(img, out_file)

        # collect predictions
        if args.save_pred:
            preds.append((frame_idx, results, track_ids, det_bboxes))

    # reverse
    if args.reverse:
        preds = preds[::-1]

    # total runtime
    runtime = time.perf_counter() - start_time
    print("\nTotal runtime: {:.6f} [s]".format(runtime))

    # save predictions
    if args.save_pred:
        basename = cvut.basename(args.video, wo_fmt=True)
        pkl_file = os.path.join(args.out_dir, "{}.pkl".format(basename))
        mmcv.dump(preds, pkl_file)
        print("Result is saved at {}".format(pkl_file))

    if args.save_video:
        print("Output video is saved at", out_video_file)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        main()
