# -- coding: utf-8 --
# @Time : 2024/1/16
# @Author : ykk648

import onnx
from onnxconverter_common import float16

model = onnx.load("")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "")
