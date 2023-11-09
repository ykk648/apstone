# -- coding: utf-8 --
# @Time : 2021/11/29
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone


import onnxruntime
import numpy as np
from cv2box import MyFpsCounter
import os
from pathlib import Path
import re


def get_output_info(onnx_session):
    output_name = []
    output_shape = []
    output_type = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
        output_shape.append(node.shape)
        output_type.append(node.type)
    return output_name, output_shape, output_type


def get_input_info(onnx_session):
    input_name = []
    input_shape = []
    input_type = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
        input_shape.append(node.shape)
        input_type.append(node.type)
    return input_name, input_shape, input_type


def get_input_feed(input_name, image_tensor):
    """
    Args:
        input_name:
        image_tensor: [image tensor, ...]
    Returns:
    """
    input_feed = {}
    for index, name in enumerate(input_name):
        input_feed[name] = image_tensor[index]
    return input_feed


class ONNXModel:
    def __init__(self, onnx_path, provider='gpu', warmup=False, debug=False, input_dynamic_shape=None):
        self.provider = provider
        try:
            onnx_name = Path(onnx_path).stem
            trt_cache_path = './cache/trt/' + onnx_name
        except TypeError:
            pass

        if self.provider == 'gpu':
            self.providers = (
                "CUDAExecutionProvider",
                {'device_id': 0, }
            )
        elif self.provider == 'trt':
            os.makedirs(trt_cache_path, exist_ok=True)
            self.providers = (
                'TensorrtExecutionProvider',
                {'trt_engine_cache_enable': True, 'trt_engine_cache_path': trt_cache_path, 'trt_fp16_enable': False, }
            )
        elif self.provider == 'trt16':
            os.makedirs(trt_cache_path, exist_ok=True)
            self.providers = (
                'TensorrtExecutionProvider',
                {'trt_engine_cache_enable': True, 'trt_engine_cache_path': trt_cache_path, 'trt_fp16_enable': True,
                 'trt_dla_enable': False, }
            )
        elif self.provider == 'trt8':
            os.makedirs(trt_cache_path, exist_ok=True)
            self.providers = (
                'TensorrtExecutionProvider',
                {'trt_engine_cache_enable': True, 'trt_int8_enable': True, }
            )
        else:
            self.providers = "CPUExecutionProvider"

        # onnxruntime.set_default_logger_severity(3)
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3

        # When env change leads to trt cache load fail, auto generate new cache
        try:
            self.onnx_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=[self.providers])
        except Exception as e:
            if type(e.args[0]) == str and 'TensorRT EP could not deserialize engine from cache' in e.args[0]:
                res = re.match('.*TensorRT EP could not deserialize engine from cache: (.*)', e.args[0])
                os.remove(res.group(1))
                print('waiting generate new model...')
                self.onnx_session = onnxruntime.InferenceSession(onnx_path, session_options, providers=[self.providers])
            else:
                raise e

        # sessionOptions.intra_op_num_threads = 3
        self.input_name, self.input_shape, self.input_type = get_input_info(self.onnx_session)
        self.output_name, self.output_shape, self.output_type = get_output_info(self.onnx_session)

        self.input_dynamic_shape = input_dynamic_shape

        if self.input_dynamic_shape is not None:
            self.input_dynamic_shape = self.input_dynamic_shape if isinstance(self.input_dynamic_shape, list) else [
                self.input_dynamic_shape]

        if debug:
            print('onnx version: {}'.format(onnxruntime.__version__))
            print("input_name:{}, \nshape:{}, \ntype:{}".format(self.input_name, self.input_shape, self.input_type))
            print("output_name:{}, \nshape:{}, \ntype:{}".format(self.output_name, self.output_shape, self.output_type))

        if warmup:
            self.warm_up()

    def warm_up(self):
        if not self.input_dynamic_shape:
            try:
                self.forward([np.random.rand(*self.input_shape[i]).astype(np.float32)
                              for i in range(len(self.input_shape))])
            except TypeError:
                print('Model may be dynamic, plz name the \'input_dynamic_shape\' !')
        else:
            self.forward([np.random.rand(*self.input_dynamic_shape[i]).astype(np.float32)
                          for i in range(len(self.input_shape))])
        print('Model warm up done !')

    def speed_test(self):
        if not self.input_dynamic_shape:
            input_tensor = [np.random.rand(*self.input_shape[i]).astype(np.float32)
                            for i in range(len(self.input_shape))]
        else:
            input_tensor = [np.random.rand(*self.input_dynamic_shape[i]).astype(np.float32)
                            for i in range(len(self.input_shape))]

        with MyFpsCounter('[{}] onnx 10 times'.format(self.provider)) as mfc:
            for i in range(10):
                _ = self.forward(input_tensor)

    def forward(self, image_tensor_in, trans=False):
        """
        Args:
            image_tensor_in: image_tensor [image_tensor] [image_tensor_1, image_tensor_2]
            trans: apply trans for image_tensor or first image_tensor(list)
        Returns:
            model output
        """
        if not isinstance(image_tensor_in, list) or len(image_tensor_in) == 1:
            image_tensor_in = image_tensor_in[0] if isinstance(image_tensor_in, list) else image_tensor_in
            if trans:
                image_tensor_in = image_tensor_in.transpose(2, 0, 1)[np.newaxis, :]
            image_tensor_in = [np.ascontiguousarray(image_tensor_in)]
        else:
            # for multi input, only trans first tensor
            if trans:
                image_tensor_in[0] = image_tensor_in[0].transpose(2, 0, 1)[np.newaxis, :]
            image_tensor_in = [np.ascontiguousarray(image_tensor) for image_tensor in image_tensor_in]

        input_feed = get_input_feed(self.input_name, image_tensor_in)
        return self.onnx_session.run(self.output_name, input_feed=input_feed)

    def batch_forward(self, bach_image_tensor, trans=False):
        if trans:
            bach_image_tensor = bach_image_tensor.transpose(0, 3, 1, 2)
        input_feed = get_input_feed(self.input_name, bach_image_tensor)
        return self.onnx_session.run(self.output_name, input_feed=input_feed)

    def close(self):
        del self.model
        self.model = None

    def cuda_binding_forward(self, image_tensor_in_list, output_type='ort'):
        """
        Args:
            image_tensor_in: torch tensor
        """
        binding = self.onnx_session.io_binding()
        for index, image_tensor_in in enumerate(image_tensor_in_list):
            # image_tensor_in = image_tensor_in.contiguous()
            if isinstance(image_tensor_in, onnxruntime.capi.onnxruntime_inference_collection.OrtValue):
                binding.bind_ortvalue_input(self.input_name[index], image_tensor_in)
            else:
                import torch
                assert isinstance(image_tensor_in, torch.Tensor)
                binding.bind_input(
                    name=self.input_name[index],
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=self.input_shape[index],
                    buffer_ptr=image_tensor_in.data_ptr(), )
        Y_tensor_list = []
        for index, ouput_name in enumerate(self.output_name):
            if output_type == 'ort':
                Y_tensor = onnxruntime.OrtValue.ortvalue_from_shape_and_type(self.output_shape[index], np.float32,
                                                                             'cuda', 0)
            else:
                import torch
                Y_tensor = torch.empty(self.output_shape[index], dtype=torch.float32, device='cuda:0').contiguous()
            binding.bind_ortvalue_output(ouput_name, Y_tensor)
            # binding.bind_output(
            #     name=ouput_name,
            #     device_type='cuda',
            #     device_id=0,
            #     element_type=np.float32,
            #     shape=self.output_shape[index],
            #     buffer_ptr=Y_tensor.data_ptr(), )
            Y_tensor_list.append(Y_tensor)
        self.onnx_session.run_with_iobinding(binding)
        return Y_tensor_list

    def binding_forward(self, binding):
        """
        ref io_binding https://onnxruntime.ai/docs/api/python/api_summary.html
        do io_binding outside this func and pass it in
        example(cuda in&out):
            import torch
            binding = self.model.onnx_session.io_binding()
            image_tensor_in = image_tensor_in.contiguous()
            binding.bind_input(
                name='X',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(image_tensor_in.shape),
                buffer_ptr=image_tensor_in.data_ptr(),)

            Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
            binding.bind_output(
                name='Y',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(Y_tensor.shape),
                buffer_ptr=Y_tensor.data_ptr(),)
        """
        Y_tensor = torch.empty(self.output_shape, dtype=torch.float32, device='cuda:0').contiguous()
        binding.bind_output(
            name=self.output_name[0],
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(Y_tensor.shape),
            buffer_ptr=Y_tensor.data_ptr(), )
        self.onnx_session.run_with_iobinding(binding)
