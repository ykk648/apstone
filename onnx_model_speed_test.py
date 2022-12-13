# -- coding: utf-8 --
# @Time : 2022/8/29
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
from apstone import ONNXModel

if __name__ == '__main__':
    onnx_p = '../AI_power/pretrain_models/sr_lib/RealESRGAN_x4plus-dynamic.onnx'
    input_dynamic_shape = (1, 3, 768, 1080)
    # input_dynamic_shape = None

    # # cpu
    # ONNXModel(onnx_p, provider='cpu', debug=True, input_dynamic_shape=input_dynamic_shape).speed_test()

    # gpu
    ONNXModel(onnx_p, provider='gpu', debug=True, input_dynamic_shape=input_dynamic_shape).speed_test()

    # # trt
    # ONNXModel(onnx_p, provider='trt', debug=False, input_dynamic_shape=input_dynamic_shape).speed_test()
    #
    # # trt16
    # ONNXModel(onnx_p, provider='trt16', debug=False, input_dynamic_shape=input_dynamic_shape).speed_test()

    # # trt int8
    # ONNXModel(onnx_p, provider='trt8', debug=False, input_dynamic_shape=input_dynamic_shape).speed_test()
