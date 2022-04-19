import numpy as np


class IoUTracker(object):

    def __init__(self,
                 match_coeff=[0.0, 1.0],
                 det_thr=0.5,
                 iou_thr=0.5,
                 track_thr=None,
                 track_age=10,
                 pre_threshold=True,
                 **kargs):

        # Properties
        self.match_coeff = match_coeff
        self.det_thr = det_thr
        self.iou_thr = iou_thr
        self.track_age = track_age
        self.pre_threshold = pre_threshold

        if track_thr is not None:
            self.track_thr = track_thr
        else:
            self.track_thr = match_coeff[0] * np.log(det_thr) \
                + match_coeff[1] * iou_thr

        # Initialize
        self.queue_bboxes = None
        self.obj_ages = None

    def __call__(self, det_bboxes, is_first=False):
        if self.pre_threshold:
            det_bboxes = self._pre_thresholding(det_bboxes)

        track_ids = self.get_obj_ids(det_bboxes, is_first)

        track_ids, det_bboxes = self._format_numpy(track_ids, det_bboxes)

        return track_ids, det_bboxes

    def _pre_thresholding(self, det_bboxes):
        selected_ids = det_bboxes[:, -1] >= self.det_thr
        det_bboxes = det_bboxes[selected_ids]
        return det_bboxes

    def _format_numpy(self, track_ids, det_bboxes):
        selected_ids = (track_ids != -1) * (det_bboxes[:, -1] >= self.det_thr)

        track_ids = track_ids[selected_ids]
        det_bboxes = det_bboxes[selected_ids]

        return track_ids, det_bboxes

    def get_obj_ids(self, det_bboxes, is_first):
        # Increase obj_ages
        if self.obj_ages is not None:
            self.obj_ages = [obj_age+1 for obj_age in self.obj_ages]

        # If no bbox
        if len(det_bboxes) == 0:
            det_obj_ids = np.array([], dtype=int)
            if is_first:
                self.queue_bboxes = None
                self.obj_ages = None
            return det_obj_ids

        # First frame of the video
        if is_first or (not is_first and self.queue_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.shape[0])
            self.queue_bboxes = det_bboxes
            self.obj_ages = [0] * len(det_bboxes)
            self.queue_ids = list(range(len(det_bboxes)))
            self.removed_ids = []

        # Compute bbox match feature
        else:
            bbox_ious = self.compute_bbox_ious(
                det_bboxes[:, :4], self.queue_bboxes[:, :4])

            # Compute comprehensive score
            comp_scores = self._compute_comp_scores(
                det_bboxes[:, 4].reshape(-1, 1), bbox_ious)
            match_ids = comp_scores.argmax(axis=1)

            # Translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            det_obj_ids = -1 * np.ones((match_ids.shape[0]), dtype=int)
            best_match_scores = self.track_thr * np.ones((
                self.queue_bboxes.shape[0]))

            for idx, match_id in enumerate(match_ids):
                # Multiple candidate might match with previous object,
                # here we choose the one with largest comprehensive score
                match_score = comp_scores[idx, match_id]

                if match_score > best_match_scores[match_id]:  # matched
                    det_obj_ids[idx] = self.queue_ids[match_id]
                    best_match_scores[match_id] = match_score
                    self.queue_bboxes[match_id] = det_bboxes[idx]
                    self.obj_ages[match_id] = 0

                else:  # add new object
                    new_obj_id = self._generate_new_obj_id()
                    det_obj_ids[idx] = new_obj_id
                    self.queue_ids.append(new_obj_id)
                    self.obj_ages.append(0)
                    self.queue_bboxes = np.concatenate([
                        self.queue_bboxes, det_bboxes[idx, None]], axis=0)

            # Remove old objects
            if self.track_age is not None:
                keep_indicators = [
                    obj_age < self.track_age for obj_age in self.obj_ages]
                self.queue_bboxes = self.queue_bboxes[keep_indicators]
                self.obj_ages = [
                    self.obj_ages[idx] for idx, keep in enumerate(
                        keep_indicators) if keep]
                self.removed_ids = [
                    self.queue_ids[idx] for idx, keep in enumerate(
                        keep_indicators) if not keep]
                self.queue_ids = [
                    self.queue_ids[idx] for idx, keep in enumerate(
                        keep_indicators) if keep]

        return det_obj_ids

    def _generate_new_obj_id(self):
        next_obj_id = max(self.queue_ids) + 1
        while True:
            if next_obj_id in self.removed_ids:
                next_obj_id += 1
            else:
                break
        return next_obj_id

    @ staticmethod
    def compute_bbox_ious(bboxes1, bboxes2):
        """
        Args:
            bboxes1 (np.float32) of shape [M, 4].
            bboxes2 (np.float32) of shape [N, 4].

        Returns:
            ious (np.float32) of shape [M, N].
        """
        rows, cols = bboxes1.shape[0], bboxes2.shape[0]
        if rows * cols == 0:
            return np.zeros([rows, cols])

        lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])

        wh = (rb - lt + 1).clip(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]

        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
            bboxes2[:, 3] - bboxes2[:, 1] + 1)

        ious = overlap / (area1[:, None] + area2 - overlap + 1e-6)
        return ious

    def _compute_comp_scores(self, bbox_scores, bbox_ious):
        """
        Args:
            bbox_scores (np.float32) of shape [M, 1]
            bbox_ious (np.float32) of shape [M, N]

        Returns:
            score (np.float32) of shape [M, N]
        """
        score = self.match_coeff[0] * np.log(bbox_scores) \
            + self.match_coeff[1] * bbox_ious
        return score


# ------------------------------------------------------------------------------
#  Testbench
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    trackor = IoUTracker()
    box1 = np.array([
        [0, 1, 12, 8],
        [0, 2, 12, 8],
    ])
    box2 = np.array([
        [0, 2, 12, 8],
        [0, 1, 12, 8],
        [0, 0, 8, 12],
    ])
    ious = trackor.compute_bbox_ious(box1, box2)
    print(ious)
