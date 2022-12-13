# -- coding: utf-8 --
# @Time : 2021/11/29
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone

import onnxruntime
import numpy as np
from cv2box import MyFpsCounter


def init_session(onnx_path, provider='gpu'):
    if provider == 'gpu':
        providers = (
            "CUDAExecutionProvider",
            {'device_id': 0, }
        )
    elif provider == 'trt':
        providers = (
            'TensorrtExecutionProvider',
            {'trt_engine_cache_enable': True, 'trt_fp16_enable': False, }
        )
    elif provider == 'trt16':
        providers = (
            'TensorrtExecutionProvider',
            {'trt_engine_cache_enable': True, 'trt_fp16_enable': True, }
        )
    elif provider == 'trt8':
        providers = (
            'TensorrtExecutionProvider',
            {'trt_engine_cache_enable': True, 'trt_int8_enable': True, }
        )
    else:
        providers = "CPUExecutionProvider"

    # onnxruntime.set_default_logger_severity(3)
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3
    onnx_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=[providers])
    return onnx_session


class OnnxModelPickable:  # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, onnx_path, provider='gpu', input_dynamic_shape=None):
        self.onnx_path = onnx_path
        self.provider = provider
        # self.onnx_session = init_session(self.onnx_path, self.provider)

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def get_output_info(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        output_shape = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
            output_shape.append(node.shape)
        return output_name, output_shape

    def get_input_info(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        input_shape = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
            input_shape.append(node.shape)
        return input_name, input_shape

    def forward(self, image_tensor, trans=False):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        if trans:
            image_tensor = image_tensor.transpose(2, 0, 1)
            image_tensor = image_tensor[np.newaxis, :]
        image_tensor = np.ascontiguousarray(image_tensor)
        input_name, _ = self.get_input_info(self.onnx_session)
        output_name, _ = self.get_output_info(self.onnx_session)
        input_feed = self.get_input_feed(input_name, image_tensor)
        return self.onnx_session.run(output_name, input_feed=input_feed)

    def __getstate__(self):
        return {
            'onnx_path': self.onnx_path,
            'provider': self.provider,
        }

    def __setstate__(self, values):
        self.onnx_path = values['onnx_path']
        self.provider = values['provider']
        self.onnx_session = init_session(self.onnx_path, self.provider)
