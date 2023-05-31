# -- coding: utf-8 --
# @Time : 2022/7/29
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone

from pathlib import Path
from .wrappers.onnx_wrapper import ONNXModel, OnnxModelPickable


class ModelBase:
    def __init__(self, model_info, provider):
        self.model_path = model_info['model_path']

        if 'input_dynamic_shape' in model_info.keys():
            self.input_dynamic_shape = model_info['input_dynamic_shape']
        else:
            self.input_dynamic_shape = None

        if 'picklable' in model_info.keys():
            picklable = model_info['picklable']
        else:
            picklable = False

        # init model
        if Path(self.model_path).suffix == '.engine':
            if 'trt_wrapper_self' in model_info.keys():
                from .wrappers.trt_wrapper import TRTWrapperSelf
                TRTWrapper = TRTWrapperSelf
            else:
                from .wrappers.trt_wrapper import TRTWrapper
            self.model_type = 'trt'
            self.model = TRTWrapper(self.model_path)
        elif Path(self.model_path).suffix == '.tjm':
            from .wrappers.tjm_wrapper import TJMWrapper
            self.model_type = 'tjm'
            self.model = TJMWrapper(self.model_path, provider=provider)
        elif Path(self.model_path).suffix in ['.onnx', '.bin']:
            self.model_type = 'onnx'
            if not picklable:
                if 'encrypt' in model_info.keys():
                    from cv2box.utils.encrypt import CVEncrypt
                    self.model_path = CVEncrypt(model_info['encrypt']).load_encrypt_file(self.model_path)
                self.model = ONNXModel(self.model_path, provider=provider,
                                              input_dynamic_shape=self.input_dynamic_shape)
            else:
                self.model = OnnxModelPickable(self.model_path, provider=provider, )

        else:
            raise 'check model suffix , support engine/tjm/onnx/bin now.'
