import cvut
import mmcv
import argparse
import numpy as np
import scipy.special


# ------------------------------------------------------------------------------
#  Utils
# ------------------------------------------------------------------------------
def compute_areas(bboxes, norm=False):
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    if norm:
        min_val = areas.min()
        max_val = areas.max()
        areas = (areas - min_val) / (max_val - min_val)
    return areas


def compute_softmax(vals, temp=1):
    return scipy.special.softmax(vals / temp)


# ------------------------------------------------------------------------------
#  ArgumentParser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--video-dir", type=str, help="Video directory",
                    default="./data/TestA/")

parser.add_argument("--pkl-dir", type=str, help="Pkl directory",
                    default="cache/cls_from_det")

parser.add_argument("--out", type=str, help="Output file",
                    default="cache/submission.txt")

parser.add_argument("--temp", type=float, help="Softmax temperature",
                    default=1.0)

parser.add_argument("--alpha", type=float, help="Alpha", default=0.5)

parser.add_argument("--beta", type=float, help="Beta", default=1)

parser.add_argument("--gamma", type=float, help="Gamma", default=1)

parser.add_argument("--min-age", type=int, help="Min age to filter noise",
                    default=5)

parser.add_argument('--check-num-box', action='store_true')

args = parser.parse_args()


# ------------------------------------------------------------------------------
#  Main execution
# ------------------------------------------------------------------------------
def main():
    # get video_files and pkl_files
    video_files = cvut.glob_files(args.video_dir, 'mp4')
    pkl_files = [f"{args.pkl_dir}/{cvut.basename(video_file, wo_fmt=True)}.pkl"
                 for video_file in video_files]

    submission_results = []
    for pkl_id, item in enumerate(zip(video_files, pkl_files)):
        video_file, pkl_file = item

        # read pred
        vid_id = pkl_id + 1
        preds = mmcv.load(pkl_file)

        # read video
        cap, (W, H), num_frames, fps = cvut.get_video(video_file)

        bbox_list, score_list, label_list, track_list, frame_list = [], [], [], [], []
        for frame_idx, results in preds:
            if results is None:
                continue

            # collect items
            det_labels = np.concatenate([
                idx * np.ones([len(result)])
                for idx, result in enumerate(results)])
            det_bboxes = np.concatenate(results)
            det_scores = det_bboxes[:, 4].copy()
            track_ids = det_bboxes[:, 5].copy()
            det_bboxes = det_bboxes[:, :4].copy()

            bbox_list.append(det_bboxes)
            score_list.append(det_scores)
            label_list.append(det_labels)
            track_list.append(track_ids)
            frame_list.append([frame_idx] * len(det_bboxes))

        # concat items
        bboxes = np.concatenate(bbox_list)
        scores = np.concatenate(score_list)
        labels = np.concatenate(label_list).astype(int)
        tracks = np.concatenate(track_list).astype(int)
        frames = np.concatenate(frame_list).astype(int)

        # analyze each track
        track_dict = dict()
        uniq_tracks = np.unique(tracks)
        for uniq_track in uniq_tracks:
            obj_mask = tracks == uniq_track
            obj_bboxes = bboxes[obj_mask]
            obj_scores = scores[obj_mask]
            obj_labels = labels[obj_mask]
            obj_frames = frames[obj_mask]

            # filter noise: the obj exists to short
            if len(obj_frames) < args.min_age:
                continue

            # get class_id
            obj_areas = compute_areas(obj_bboxes, norm=True)
            label_scores = compute_softmax(
                (obj_areas**args.beta) * (obj_scores**args.gamma),
                temp=args.temp)
            uniq_obj_labels, count = np.unique(obj_labels, return_counts=True)
            for idx in range(len(label_scores)):
                freq = count[uniq_obj_labels == obj_labels[idx]]
                label_scores[idx] *= args.alpha * freq
            vote_label = obj_labels[np.argmax(label_scores)]
            class_id = vote_label + 1

            # get timestamp
            frame = obj_frames[len(obj_frames)//2]
            timestamp = int(frame / fps)

            # collect result
            # submission_results.append((vid_id, class_id, timestamp))
            track_dict[uniq_track] = dict(label=class_id,
                                          bboxes=obj_bboxes,
                                          frames=obj_frames,
                                          vid_id=vid_id,
                                          timestamp=timestamp)

        # sort track by timestamp
        track_dict = dict(sorted(track_dict.items(),
                                 key=lambda item: item[1]['timestamp']))

        # merge duplicated tracks
        num_objs = len(track_dict)
        track_ids = list(track_dict.keys())
        removed_track_ids = []
        for idx in range(len(track_dict)):

            # get two consecutive tracks
            if idx == num_objs - 1:
                continue
            curr_track_id = track_ids[idx]
            next_track_id = track_ids[idx+1]
            curr_track = track_dict[curr_track_id]
            next_track = track_dict[next_track_id]

            # check whether they share the same label
            if curr_track['label'] != next_track['label']:
                continue

            # check whether their timestamp are consecutive
            if np.abs(curr_track['timestamp'] - next_track['timestamp']) > 1:
                continue

            # check whether they do not exists in the same frame
            if args.check_num_box and len(list(
                    set(curr_track['frames'])
                    &
                    set(next_track['frames']))) > 1:
                continue

            # if they may duplicate, remove the current track
            removed_track_ids.append(curr_track_id)

        # remove duplicated tracks
        for idx in range(len(track_dict)):
            track_id = track_ids[idx]
            if track_id in removed_track_ids:
                track_dict.pop(track_id)

        # collect submission_results
        for track_id, attribs in track_dict.items():
            submission_results.append(
                (attribs['vid_id'], attribs['label'], attribs['timestamp']))

    # write to txt file
    with open(args.out, 'w') as fp:
        for vid_id, class_id, timestamp in submission_results:
            content = f"{vid_id} {class_id} {timestamp}\n"
            fp.writelines(content)
    print(f"Submission output is saved at {args.out}")


if __name__ == '__main__':
    main()
