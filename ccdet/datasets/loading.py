import cv2
import os.path as osp
import mmcv
import numpy as np
from glob import glob
import cvut
import random
from mmdet.datasets import PIPELINES
from PIL import Image
from mmdet.core import BitmapMasks
import cvut
import os


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1]*value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2]*value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def gausian_blur(image, blur):
    return cv2.GaussianBlur(image, (5, 5), blur)


def averageing_blur(image, shift):
    return cv2.blur(image, (shift, shift))


def median_blur(image, shift):
    return cv2.medianBlur(image, shift)


def bileteralBlur(image, d, color, space):
    return cv2.bilateralFilter(image, d, color, space)


def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    return image


def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image = cv2.LUT(image, table)
    return image


def contrast_image(image, contrast):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(
        pixel + contrast, 255) for pixel in row] for row in image[:, :, 2]]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def rotate_image(image, deg):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image


@PIPELINES.register_module()
class CopyObjectsToBackgroundImage(object):
    """Copy objects to background image"""

    def __init__(self, min_area_ratio, max_area_ratio,
                 img_foreground_dir, mask_foregroud_dir,
                 max_paste_objets=10, min_paste_objects=3,
                 center_paste=False, with_mask=False,
                 previous_bbox_mask_in_results=False,
                 label_value=-1,
                 gt_mask_ratio_path=None):

        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_paste_objets = max_paste_objets
        self.min_paste_objects = min_paste_objects
        self.with_mask = with_mask
        self.previous_bbox_mask_in_results = previous_bbox_mask_in_results
        self.label_value = label_value
        
        self.load_gt_mask_ratios = False
        # load mask ratio(mask area divided by largest mask of each class)
        if gt_mask_ratio_path:
            self.load_gt_mask_ratios = True
            print("loading gt mask ratios...")
            self.gt_ratios = mmcv.load(gt_mask_ratio_path)

        all_imgs = sorted(glob(osp.join(img_foreground_dir, '*.jpg')))
        all_masks = []
        for img_dir in all_imgs:
            all_masks.append(osp.join(mask_foregroud_dir, osp.basename(
                img_dir).replace('.jpg', '_seg.jpg')))
        self.img_mask_pairs = []
        for img, mask in zip(all_imgs, all_masks):
            # test removing low resolution image
            W, H = Image.open(img).size
            min_size = 100
            if (W < min_size) and (H < min_size):
                # print(f"low resoluton image: {H}-{W}")
                continue
            self.img_mask_pairs.append([img, mask])
        print(f"==== ignore:{len(all_imgs)-len(self.img_mask_pairs)}, \
            remaining image:{len(self.img_mask_pairs)}")
        self.center_paste = center_paste

    def __call__(self, results):
        # get imgs containing fgs (list)
        img_fgs = []
        num_objects = random.randint(
            self.min_paste_objects, self.max_paste_objets)

        img_mask_dirs = random.sample(self.img_mask_pairs, num_objects)
        img_foground_files = []
        mask_foground_files = []
        for img_info, mask_info in img_mask_dirs:
            img_foground_files.append(img_info)
            mask_foground_files.append(mask_info)
        for file in img_foground_files:
            img_fgs.append(cv2.imread(file))

        gt_masks = []
        for file in mask_foground_files:
            mask = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            mask[mask < 125] = 0
            gt_masks.append(mask)

        gt_mask_ratios = []
        if self.load_gt_mask_ratios:
            for filename in mask_foground_files:
                basename = osp.basename(filename)
                gt_mask_ratios.append(int(self.gt_ratios[basename] * 100))

        # apply augmentation directly
        augment = True
        if augment:
            # Brightness
            p_brightness = random.random()

            if p_brightness < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = brightness(img_fgs[idx], 0.5, 3)

            # horizontal flip
            p_hflip = random.random()
            if p_hflip < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = cv2.flip(img_fgs[idx], 1)
                        gt_masks[idx] = cv2.flip(gt_masks[idx], 1)

            # vertical flip
            p_vflip = random.random()
            if p_vflip < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = cv2.flip(img_fgs[idx], 0)
                        gt_masks[idx] = cv2.flip(gt_masks[idx], 0)
            # blurry
            p_bileteralBlur = random.random()
            if p_bileteralBlur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = bileteralBlur(img_fgs[idx], 40, 75, 75)
            p_averageing_blur = random.random()
            if p_averageing_blur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = averageing_blur(img_fgs[idx], 5)

            p_median_blur = random.random()
            if p_median_blur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = median_blur(img_fgs[idx], 5)

            # p_contrast = random.random()
            # if p_contrast < 0.3:
            #     for idx in range(len(img_fgs)):
            #         if random.random() < 0.5:
            #             img_fgs[idx] = contrast_image(img_fgs[idx], np.random.uniform(1, 10))

            # p_rotate = random.random()
            # if p_rotate < 0.3:
            #     for idx in range(len(img_fgs)):
            #         if random.random() < 0.5:
            #             deg = np.random.randint(10, 90)
            #             img_fgs[idx] = rotate_image(img_fgs[idx], deg)
            #             gt_masks[idx] = rotate_image(gt_masks[idx], deg)

            p_light = random.random()
            if p_light < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = add_light(img_fgs[idx], np.random.choice(
                            [0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.5]))
                        # img_fgs[idx] = add_light(img_fgs[idx], np.random.uniform(0.9, 5)) # not good

                # p_light = random.random()
            # if p_light< 0.3:
            #     img_bg = add_light(img_bg)
        gt_labels = []
        for filename in img_foground_files:
            class_id = int(osp.basename(filename).split('_')[0]) - 1
            gt_labels.append(class_id)

        # copy objects from fg to bg img
        img_bg, gt_bboxes, gt_labels, instance_masks \
            = self._copy_objs_to_bg_img(results['img'], img_fgs, gt_labels, gt_masks)

        # augment for whole image
        # p_light = random.random()
        # if p_light< 0.3:
        #     img_bg = add_light(img_bg)

        # p_light_color = random.random()
        # if (p_light_color< 0.5) and (p_light >= 0.3):
        #     img_bg = add_light_color(200, img_bg)
        # img_bg = add_light_color(img_bg, 200, 2)
        # img_bg = add_light(img_bg, 0.5)
        if self.label_value != -1:
            gt_labels = np.clip(gt_labels, 0, self.label_value)
        results['img'] = img_bg
        if not self.previous_bbox_mask_in_results:
            results['gt_bboxes'] = gt_bboxes.astype(np.float32)
            results['gt_labels'] = gt_labels
            if self.with_mask:
                results['gt_masks'] = instance_masks
                if 'mask_fields' not in results:
                    results['mask_fields'] = []
                results['mask_fields'].append('gt_masks')
            results['img'] = img_bg
        else:
            results['gt_bboxes'] = np.concatenate(
                [results['gt_bboxes'], gt_bboxes.astype(np.float32)])
            results['gt_labels'] = np.concatenate(
                [results['gt_labels'], gt_labels])
            results['gt_masks'] = BitmapMasks(
                np.concatenate([results['gt_masks'], instance_masks]),
                instance_masks.height, instance_masks.width)
        
        # borrow gt_labels to store gt mask ratios, 
        # used to supervise mask ratio prediction
        if self.load_gt_mask_ratios:
            results['gt_labels'] = gt_mask_ratios
            
        return results

    def _read_bg_img(self, results):
        bg_img_file = cv2.imread(results['bg_img_file'])
        results['img'] = bg_img_file
        results['img_shape'] = bg_img_file.shape
        results['ori_shape'] = bg_img_file.shape
        results['img_fields'] = ['img']
        return results

    def _copy_objs_to_bg_img(self, img_bg, img_fgs,
                             gt_labels, gt_masks):
        # resize img_fg, gt_bboxes, and gt_masks into img_bg size
        H, W = img_bg.shape[:2]
        w_scales = []
        h_scales = []
        for idx, img_fg in enumerate(img_fgs):
            img_fgs[idx], w_scale, h_scale = mmcv.imresize(img_fg, (W, H),
                                                           return_scale=True)
            h_scales.append(h_scale)
            w_scales.append(w_scale)

        # resize gt_masks
        gt_masks = [mmcv.imresize(gt_mask, (W, H), interpolation='nearest')
                    for gt_mask in gt_masks]

        # copy objs from fg to bg
        new_gt_bboxes = []
        new_gt_labels = []
        new_gt_instance_masks = []
        for idx, (gt_label, gt_mask) in enumerate(zip(
                gt_labels, gt_masks)):
            _img_fg = img_fgs[idx]
            _height, _width = _img_fg.shape[:2]
            x1, y1, x2, y2 = 0, 0, _width, _height
            h, w = y2 - y1, x2 - x1
            area_ratio = w * h / (W * H)

            scale = 1
            if area_ratio < self.min_area_ratio:
                continue
            if area_ratio > self.max_area_ratio:
                scale = np.sqrt(np.random.uniform(
                    self.min_area_ratio, self.max_area_ratio) / area_ratio)
                x1, y1, x2, y2, gt_mask, _img_fg = self._resize_fg(
                    x1, y1, x2, y2, gt_mask, _img_fg, scale)
                h, w = y2 - y1, x2 - x1
            img_obj = _img_fg[y1:y2, x1:x2]
            mask_obj = gt_mask[y1:y2, x1:x2].astype(bool)
            xp, yp = self._sample_loc_to_paste(H, W, h, w)
            img_bg[yp:yp+h, xp:xp+w][mask_obj] = img_obj[mask_obj]
            if self.with_mask:
                _new_instance_mask = np.zeros((H, W), dtype=np.uint8)
                _new_instance_mask[yp:yp+h, xp:xp+w] = mask_obj
                new_gt_instance_masks.append(_new_instance_mask)
            new_gt_bboxes.append((xp, yp, xp+w, yp+h))
            new_gt_labels.append(gt_label)
        new_gt_bboxes = np.array(new_gt_bboxes)
        new_gt_labels = np.array(new_gt_labels)
        new_gt_instance_masks = np.stack(new_gt_instance_masks, 0)
        new_gt_instance_masks = BitmapMasks(new_gt_instance_masks, H, W)
        return img_bg, new_gt_bboxes, new_gt_labels, new_gt_instance_masks

    def _resize_fg(self, x1, y1, x2, y2, gt_mask, img_fg, scale):
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gt_mask = mmcv.imrescale(gt_mask, scale, interpolation='nearest')
        img_fg = mmcv.imrescale(img_fg, scale)
        return x1, y1, x2, y2, gt_mask, img_fg

    def _sample_loc_to_paste(self, H, W, h, w):
        assert (h <= H) and (w <= W)
        if self.center_paste:
            center_x, center_y = W//2, H//2
            xp = int(np.random.normal(center_x, W//8))
            yp = int(np.random.normal(center_y, H//8))
        else:
            xp = random.randint(0, W-w) if W != w else 0
            yp = random.randint(0, H-h) if H != h else 0
        if xp + w >= W or yp+h >= H or xp < 0 or yp < 0:
            xp = random.randint(0, W-w) if W != w else 0
            yp = random.randint(0, H-h) if H != h else 0
        return xp, yp


@PIPELINES.register_module()
class RandomGenerateBGImageGAN(object):
    def __init__(self,
                 width=1920,
                 height=1080, gan_img_path=[]):
        self.width = width
        self.height = height
        self.gan_img_path = []
        for fl in gan_img_path:
            self.gan_img_path.extend([os.path.join(
                fl, image_name) for image_name in os.listdir(fl)])

    def __call__(self, results):
        if np.random.rand() < 0.3:
            value = random.randint(120, 220)
            img = np.zeros([self.height, self.width, 3], dtype=np.uint8)
            # if random.random()< 0.5:
            #     img = np.random.normal(255./2,255./10,img.shape)
            # else:
            noise = np.random.normal(255./2, 255./10, img.shape)
            img[:] = value
            H, W = img.shape[:2]
            img[H-300:, ...] = noise[H-300:, ...]
        else:
            np.random.shuffle(self.gan_img_path)
            bg = cv2.imread(self.gan_img_path[0])
            old_size = bg.shape[:2]  # old_size is in (height, width) format

            ratio_width = float(self.width)/old_size[1]
            ratio_height = float(self.height)/old_size[0]

            # new_size = tuple([int(x*ratio) for x in old_size])
            new_width = old_size[1]*int(ratio_width)
            new_height = old_size[0]*int(ratio_height)

            # new_size should be in (width, height) format

            # import ipdb
            # ipdb.set_trace()
            im = cv2.resize(bg, (new_width, new_height))

            delta_w = self.width - new_width
            delta_h = self.height - new_height
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class RandomGenerateBGImage(object):
    def __init__(self,
                 width=1920,
                 height=1080):
        self.width = width
        self.height = height

    def __call__(self, results):
        value = random.randint(120, 220)
        img = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        # if random.random()< 0.5:
        #     img = np.random.normal(255./2,255./10,img.shape)
        # else:
        noise = np.random.normal(255./2, 255./10, img.shape)
        img[:] = value
        H, W = img.shape[:2]
        img[H-300:, ...] = noise[H-300:, ...]
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class RandomCopyPaste(CopyObjectsToBackgroundImage):
    """Copy objects to background image, no annotations"""

    def __init__(self, min_area_ratio, max_area_ratio,
                 img_foreground_dir, center_paste=False,
                 max_paste_objets=10, min_paste_objects=3,
                 keep_bbox_masks=False, label_value=1,
                 previous_bbox_mask_in_results=False):

        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_paste_objets = max_paste_objets
        self.min_paste_objects = min_paste_objects

        self.all_imgs = sorted(cvut.glob_imgs(
            img_foreground_dir, recursive=True))
        self.center_paste = center_paste
        self.keep_bbox_masks = keep_bbox_masks
        self.label_value = label_value
        self.previous_bbox_mask_in_results = previous_bbox_mask_in_results

    def __call__(self, results):
        # get imgs containing fgs (list)
        img_fgs = []
        num_objects = random.randint(
            self.min_paste_objects, self.max_paste_objets)

        img_dirs = random.sample(self.all_imgs, num_objects)

        for file in img_dirs:
            img_fgs.append(cv2.imread(file))

        # apply augmentation directly
        augment = True
        if augment:
            # Brightness
            p_brightness = random.random()

            if p_brightness < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = brightness(img_fgs[idx], 0.5, 3)

            # horizontal flip
            p_hflip = random.random()
            if p_hflip < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = cv2.flip(img_fgs[idx], 1)

            # vertical flip
            p_vflip = random.random()
            if p_vflip < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = cv2.flip(img_fgs[idx], 0)

            # blurry
            p_bileteralBlur = random.random()
            if p_bileteralBlur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = bileteralBlur(img_fgs[idx], 40, 75, 75)

            p_averageing_blur = random.random()
            if p_averageing_blur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = averageing_blur(img_fgs[idx], 5)

            p_median_blur = random.random()
            if p_median_blur < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = median_blur(img_fgs[idx], 5)

            p_light = random.random()
            if p_light < 0.3:
                for idx in range(len(img_fgs)):
                    if random.random() < 0.5:
                        img_fgs[idx] = add_light(img_fgs[idx], np.random.choice(
                            [0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.5]))

        # copy objects from fg to bg img
        img_bg, gt_bboxes, gt_labels, instance_masks \
            = self._copy_objs_to_bg_img(results['img'], img_fgs)

        if not self.previous_bbox_mask_in_results and self.keep_bbox_masks:
            results['gt_bboxes'] = gt_bboxes.astype(np.float32)
            results['gt_labels'] = gt_labels

            if self.keep_bbox_masks:
                results['gt_masks'] = instance_masks
                if 'mask_fields' not in results:
                    results['mask_fields'] = []
                results['mask_fields'].append('gt_masks')
            results['img'] = img_bg
        elif self.keep_bbox_masks:
            results['gt_bboxes'] = np.concatenate(
                results['gt_bboxes'], gt_bboxes.astype(np.float32))
            results['gt_labels'] = np.concatenate(
                results['gt_labels'], gt_labels)
            results['gt_masks'] = BitmapMasks(
                np.concatenate(results['gt_masks'], instance_masks),
                instance_masks.height, instance_masks.width)
        return results

    def _copy_objs_to_bg_img(self, img_bg, img_fgs):
        # resize img_fg, gt_bboxes, and gt_masks into img_bg size
        H, W = img_bg.shape[:2]
        w_scales = []
        h_scales = []
        for idx, img_fg in enumerate(img_fgs):
            img_fgs[idx], w_scale, h_scale = mmcv.imresize(img_fg, (W, H),
                                                           return_scale=True)
            h_scales.append(h_scale)
            w_scales.append(w_scale)

        # copy objs from fg to bg
        new_gt_labels = []
        new_gt_instance_masks = []
        new_gt_bboxes = []
        for idx in range(len(img_fgs)):
            _img_fg = img_fgs[idx]
            _height, _width = _img_fg.shape[:2]
            x1, y1, x2, y2 = 0, 0, _width, _height
            h, w = y2 - y1, x2 - x1
            area_ratio = w * h / (W * H)
            scale = 1
            if area_ratio < self.min_area_ratio:
                continue
            if area_ratio > self.max_area_ratio:
                scale = np.sqrt(np.random.uniform(
                    self.min_area_ratio, self.max_area_ratio) / area_ratio)
                x1, y1, x2, y2, _img_fg = self._resize_fg(
                    x1, y1, x2, y2, _img_fg, scale)
                h, w = y2 - y1, x2 - x1
            img_obj = _img_fg[y1:y2, x1:x2]
            mask = img_obj.mean(-1) > 10
            xp, yp = self._sample_loc_to_paste(H, W, h, w)
            img_bg[yp:yp+h, xp:xp+w][mask] = img_obj[mask]
            if self.keep_bbox_masks:
                _new_instance_mask = np.zeros((H, W), dtype=np.uint8)
                mask = mask.astype(np.uint8)
                _new_instance_mask[yp:yp+h, xp:xp+w] = mask
                new_gt_instance_masks.append(_new_instance_mask)
                # find bbox
                m_h, m_w = mask.shape
                y1 = np.argmax(mask, 0)
                y1 = y1[y1 > 0]
                if len(y1):
                    y1 = y1.min()
                else:
                    y1 = 0

                x1 = np.argmax(mask, 1)
                x1 = x1[x1 > 0]
                if len(x1):
                    x1 = x1.min()
                else:
                    x1 = 0

                y2 = np.argmax(np.flip(mask, 0), 0)
                y2 = y2[y2 > 0]
                if len(y2):
                    y2 = y2.min()
                else:
                    y2 = 0
                y2 = m_h - y2

                x2 = np.argmax(np.flip(mask, 1), 0)
                x2 = x2[x2 > 0]
                if len(x2):
                    x2 = x2.min()
                else:
                    x2 = 0
                x2 = m_w - x2
                new_gt_bboxes.append((xp+x1, yp+y1, xp+x2, yp+y2))

        if self.keep_bbox_masks:
            new_gt_bboxes = np.array(new_gt_bboxes)
            new_gt_labels = np.ones((len(new_gt_bboxes)),
                                    dtype=int)*self.label_value
            new_gt_instance_masks = np.stack(new_gt_instance_masks, 0)
            new_gt_instance_masks = BitmapMasks(new_gt_instance_masks, H, W)
        return img_bg, new_gt_bboxes, new_gt_labels, new_gt_instance_masks

    def _resize_fg(self, x1, y1, x2, y2, img_fg, scale):
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_fg = mmcv.imrescale(img_fg, scale)
        return x1, y1, x2, y2, img_fg
