# -- coding: utf-8 --
# @Time : 2022/7/28
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
"""
mmdet model infer wrapper
"""
from cv2box import CVImage, try_import
import numpy as np
import cv2
from ..mmlab_wrapper.utils import _taylor, _gaussian_blur, _get_max_preds
transforms = try_import('torchvision.transforms', '')
from ...model_base import ModelBase


class KpDetectorBase(ModelBase):
    def __init__(self, model_info, provider):
        super().__init__(model_info, provider)
        self.dark_flag = None
        self.ratio = self.left = self.top = None
        if 'model_input_size' in model_info.keys():
            self.model_input_size = model_info['model_input_size']
        if 'kernel' in model_info.keys():
            self.kernel_size = model_info['kernel']
        else:
            self.kernel_size = 11

    def trans_pred(self, preds, ratio, left, top, maxvals, H):
        kp_results = []
        for index, kp in enumerate(preds[0]):
            x, y = preds[0][index]
            # 4 / ratio
            new_ratio = (self.model_input_size[1] // H) / ratio
            new_y = y * new_ratio + top
            new_x = x * new_ratio + left
            kp_results.append([new_x, new_y, float(maxvals[0][index][0])])
        # print(kp_results)
        return kp_results

    def post_process_default(self, heatmaps, ratio, left, top):
        # postprocess
        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)
        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 <= px <= W - 2 and 1 <= py <= H - 2:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25
        return self.trans_pred(preds, ratio, left, top, maxvals, H)

    def post_process_modelscope(self, heatmaps, ratio, left, top):
        N, K, H, W = heatmaps.shape
        _, maxvals = _get_max_preds(heatmaps)
        preds = []
        scores = []
        for i in range(K):
            heatmap = heatmaps[i, :, :]
            pt = np.where(heatmap == np.max(heatmap))
            scores.append(np.max(heatmap))
            x = pt[1][0]
            y = pt[0][0]

            if x >= 1 and x <= W - 2 and y >= 1 and y <= H - 2:
                x_diff = heatmap[y, x + 1] - heatmap[y, x - 1]
                y_diff = heatmap[y + 1, x] - heatmap[y - 1, x]
                x_sign = 0
                y_sign = 0
                if x_diff < 0:
                    x_sign = -1
                if x_diff > 0:
                    x_sign = 1
                if y_diff < 0:
                    y_sign = -1
                if y_diff > 0:
                    y_sign = 1
                x = x + x_sign * 0.25
                y = y + y_sign * 0.25

            preds.append([x, y])
        return self.trans_pred(preds, ratio, left, top, maxvals, H)

    def post_process_unbiased(self, heatmaps, ratio, left, top, kernel=11):
        # apply Gaussian distribution modulation.
        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)
        heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        for n in range(N):
            for k in range(K):
                preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        return self.trans_pred(preds, ratio, left, top, maxvals, H)

    def preprocess(self, image_in_, bbox_, mirror=False):
        img_resize, self.ratio, self.left, self.top = CVImage(image_in_).crop_keep_ratio(bbox_, self.model_input_size)
        if mirror:
            img_resize = cv2.flip(img_resize, 1)
        if self.model_type == 'trt':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            blob = dict(input=CVImage(img_resize).set_transform(transform).tensor().cuda())
        else:
            blob = CVImage(img_resize).innormal(mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.1200, 57.375]).transpose(2, 0, 1)[np.newaxis, :]
        return blob

    # def preprocess_batch(self, image_in_batch_, bbox_batch_, mirror=False):
    #     img_resize, self.ratio, self.left, self.top = CVImage(image_in_).crop_keep_ratio(bbox_, self.model_input_size)
    #     if mirror:
    #         img_resize = cv2.flip(img_resize, 1)
    #     if self.model_type == 'trt':
    #         transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #         ])
    #         blob = dict(input=CVImage(img_resize).set_transform(transform).tensor().cuda())
    #     else:
    #         blob = CVImage(img_resize).innormal(mean=[123.675, 116.28, 103.53],
    #                                             std=[58.395, 57.1200, 57.375], to_rgb=False).transpose(2, 0, 1)[np.newaxis, :]
    #     return blob

    def postprocess(self, model_results):
        if self.model_type == 'trt':
            heatmaps = model_results['output'].cpu().numpy()
        else:
            heatmaps = model_results[0]
        if self.dark_flag:
            kp_results = self.post_process_unbiased(heatmaps, self.ratio, self.left, self.top, kernel=self.kernel_size)
        else:
            kp_results = self.post_process_default(heatmaps, self.ratio, self.left, self.top)
        return kp_results

    def postprocess_mirror(self, model_results, model_mirror_results, flip_pairs):
        if self.model_type == 'trt':
            heatmaps = model_results['output'].cpu().numpy()
            heatmaps_mirror = model_mirror_results['output'].cpu().numpy()
        else:
            heatmaps = model_results[0]
            heatmaps_mirror = model_mirror_results[0]

        heatmaps_mirror_back = heatmaps_mirror.copy()
        # # Swap left-right parts
        for left_id, right_id in flip_pairs:
            heatmaps_mirror_back[:, left_id, ...] = heatmaps_mirror[:, right_id, ...]
            heatmaps_mirror_back[:, right_id, ...] = heatmaps_mirror[:, left_id, ...]
        heatmaps_mirror_back = heatmaps_mirror_back[..., ::-1]
        heatmaps = (heatmaps + heatmaps_mirror_back) * 0.5

        if self.model_type.find('dark') > 0:
            kp_results = self.post_process_default(heatmaps, self.ratio, self.left, self.top)
        else:
            kp_results = self.post_process_unbiased(heatmaps, self.ratio, self.left, self.top)
        return kp_results

    def show(self, image_in_, results_):
        img_origin = CVImage(image_in_).bgr.copy()
        img_show = CVImage(img_origin).draw_landmarks(results_)
        CVImage(img_show).show(0, "human kp result")
