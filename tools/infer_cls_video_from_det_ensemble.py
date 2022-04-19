import os
import cv2
import time
import mmcv
import cvut
import torch
import argparse
import numpy as np


from cccls.datasets import build_dataset
from ccdet.apis import draw_det_result, draw_track_result
from cccls.apis import build_cls_data_pipeline, init_classifier, get_cls_data


# ------------------------------------------------------------------------------
#  ArgumentParser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--cfg', type=str, nargs='+', help='Config file')

parser.add_argument("--ckpt", type=str, nargs='+', help="Checkpoint file")

parser.add_argument("--weight", type=float, nargs='+', help="Ensemble weights")

parser.add_argument("--video", type=str, help="Video file")

parser.add_argument("--pkl", type=str, help="Pkl file (detection output)")

parser.add_argument("--fp16", action='store_true', default=False,
                    help="Enable FP16")

parser.add_argument("--tta", action='store_true', default=False,
                    help="Enable TTA")

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--det-thr", type=float, default=0.3,
                    help="Detection threshold")

parser.add_argument("--cls-thr", type=float, default=0.5,
                    help="Classification threshold")

parser.add_argument('--roi', type=float, nargs='+', default=[0.25, 0.75],
                    help='ROI ratio (min, max) over image shape')

parser.add_argument("--start-frame", type=int, default=0,
                    help="Start frame for visualization")

parser.add_argument("--max-frames", type=int, default=-1,
                    help="Number of maximum frames for visualization")

parser.add_argument("--out-dir", type=str, default="cache/cls_from_det/",
                    help="Output directory to save visualization results")

parser.add_argument("--save-img", action='store_true', default=False,
                    help="Save image")

parser.add_argument("--save-pred", action='store_true', default=False,
                    help="Save prediction")

parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=mmcv.DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')

args = parser.parse_args()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
def main():

    # get cap
    cap, (width, height), num_frames, fps = cvut.get_video(args.video)
    print(f"[Video] FPS: {fps}; Resolution: {(width, height)}")

    out_video_file = os.path.join(args.out_dir, os.path.basename(args.video))
    out_video_file = out_video_file.replace('.mp4', '_out.mp4')
    video_shape = (width, height)
    out = cvut.create_video(out_video_file, video_shape, fps=fps)

    # build data_pipeline and model
    assert len(args.cfg) == len(args.ckpt) == len(args.weight)
    models, data_pipelines = [], []
    for cfg_file, ckpt_file in zip(args.cfg, args.ckpt):
        cfg = mmcv.Config.fromfile(cfg_file)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        data_pipeline = build_cls_data_pipeline(cfg.test_pipeline)
        model = init_classifier(cfg_file, ckpt_file,
                                device=args.device, fp16=args.fp16)
        dataset = build_dataset(cfg.data.val)
        CLASSES = dataset.CLASSES
        model.CLASSES = CLASSES + ("other", )
        models.append(model)
        data_pipelines.append(data_pipeline)

    # get start and stop frame
    start_frame = args.start_frame if args.start_frame != -1 else 0
    stop_frame = (start_frame + args.max_frames if args.max_frames != -1
                  else num_frames)

    # read pkl
    det_results = mmcv.load(args.pkl)

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
        status, image = cap.read()
        if not status:
            break
        H, W = image.shape[:2]

        # preprocessing
        det_frame_idx, det_result, track_ids, det_bboxes = \
            det_results[frame_idx]
        valid_inds = det_bboxes[:, -1] > args.det_thr
        track_ids = track_ids[valid_inds, None]
        det_bboxes = det_bboxes[valid_inds]

        det_bboxes[:, [0, 2]] = np.clip(det_bboxes[:, [0, 2]], 0, W)
        det_bboxes[:, [1, 3]] = np.clip(det_bboxes[:, [1, 3]], 0, H)

        cx = 0.5 * (det_bboxes[:, 2] + det_bboxes[:, 0])
        cy = 0.5 * (det_bboxes[:, 3] + det_bboxes[:, 1])
        inroi_inds = ((cx >= args.roi[0] * W) * (cx <= args.roi[1] * W)
                      *
                      (cy >= args.roi[0] * H) * (cy <= args.roi[1] * H))
        det_bboxes = det_bboxes[inroi_inds]
        track_ids = track_ids[inroi_inds]

        runtime = 0
        results = None
        if len(det_bboxes) > 0:

            # get data
            patches = [image[y1:y2, x1:x2]
                       for (x1, y1, x2, y2) in det_bboxes[:, :4].astype(int)]

            tic = time.perf_counter()

            scores = []
            for model, data_pipeline in zip(models, data_pipelines):
                data = get_cls_data(patches, data_pipeline, device=args.device)

                # tta
                if args.tta:
                    img = data['img'].clone()
                    img_hflip = torch.flip(img, dims=(3, ))
                    data['img'] = [img, img_hflip]

                # model inference
                score = model(return_loss=False, **data)
                score = np.stack(score)
                scores.append(score)

            # ensemble
            scores = np.stack(scores)
            weights = np.array(args.weight)[:, None, None]
            scores = (weights * scores).sum(axis=0)
            pred_scores = np.max(scores, axis=1)
            pred_labels = np.argmax(scores, axis=1)
            cls_results = []
            for pred_score, pred_label in zip(pred_scores, pred_labels):
                pred_label = int(pred_label)
                pred_score = float(pred_score)
                result = {'pred_label': pred_label,
                          'pred_score': pred_score,
                          'pred_class': model.CLASSES[pred_label]}
                cls_results.append(result)

            runtime = time.perf_counter() - tic

            # draw result
            pred_cls_ids = []
            results = [list() for _ in range(len(CLASSES))]
            for idx, cls_result in enumerate(cls_results):
                cls_idx = cls_result['pred_label']
                cls_score = cls_result['pred_score']
                if cls_idx >= 116:  # other class
                    continue
                if cls_score < args.cls_thr:
                    continue
                det_bbox = np.concatenate([det_bboxes[idx:idx+1],
                                           np.ones([1, 1]),
                                           track_ids[idx:idx+1]], axis=1)
                results[cls_idx].append(det_bbox)
                pred_cls_ids.append(cls_idx)
            pred_cls_ids = np.array(pred_cls_ids)
            for idx, result in enumerate(results):
                if len(result):
                    results[idx] = np.concatenate(result)
                else:
                    if track_ids is not None:
                        results[idx] = np.zeros([0, 6])
                    else:
                        results[idx] = np.zeros([0, 5])

            if track_ids is None:
                image = draw_det_result(results, image,
                                        CLASSES, args.det_thr)[0]
            else:
                image = draw_track_result(track_ids, det_bboxes, pred_cls_ids,
                                          image, CLASSES)[0]

        print(f"[{frame_idx+1}/{num_frames}] runtime: {int(1e3*runtime)} [ms]")

        # write result image
        if args.save_img:
            out_file = os.path.join(
                args.out_dir, f"frame_{str(frame_idx).zfill(8)}.jpg")
            cvut.imwrite(image, out_file)
        out.write(image)

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

    print("Output video is saved at", out_video_file)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        main()
