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
transforms = try_import('torchvision.transforms', '')
from ...model_base import ModelBase


class BboxDetectorBase(ModelBase):
    def __init__(self, model_info, provider):
        super().__init__(model_info, provider)
        self.ratio = self.pad_w = self.pad_h = None
        self.model_input_size = model_info['model_input_size']

    def preprocess(self, image_in_):
        img_resize, self.ratio, self.pad_w, self.pad_h = CVImage(image_in_).resize_keep_ratio(self.model_input_size,
                                                                                              pad_value=(
                                                                                                  114.0, 114.0, 114.0))
        if self.model_type == 'trt':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            blob = dict(input=CVImage(img_resize.astype(np.float32)).set_transform(transform).tensor().cuda())
        else:
            blob = img_resize.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :]
        return blob

    def postprocess(self, model_results, threshold, label=0, max_bbox_num=99):
        if self.model_type == 'trt':
            dets = model_results['dets'].cpu().numpy()
            labels = model_results['labels'].cpu().numpy()
        else:
            dets = model_results[0]
            labels = model_results[1]
        results = []
        for index, bbox in enumerate(dets[0]):
            if labels[0][index] == label and bbox[4] > threshold:
                results.append(
                    np.concatenate((CVImage(None).recover_from_resize([bbox[:2]], self.ratio, self.pad_w, self.pad_h)[0],
                                    CVImage(None).recover_from_resize([bbox[2:4]], self.ratio, self.pad_w, self.pad_h)[
                                        0])).astype(int))
        return results[:max_bbox_num]

    def show(self, image_in_, results_):
        img_origin = CVImage(image_in_).bgr.copy()
        for bbox_ in results_:
            cv2.rectangle(img_origin, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (0, 255, 0), 4)
        CVImage(img_origin).show(0, "bbox result")
        return img_origin
