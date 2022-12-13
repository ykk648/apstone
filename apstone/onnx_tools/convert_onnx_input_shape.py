# -- coding: utf-8 --
# @Time : 2022/7/27
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
"""
pip install -U onnx \
&& pip install -U sio4onnx
"""
from sio4onnx import io_change

estimated_graph = io_change(
    input_onnx_file_path="pretrain_models/hand_lib/hand_detector_yolox/yolox_100DOH_epoch90.onnx",
    output_onnx_file_path="pretrain_models/hand_lib/hand_detector_yolox/yolox_100DOH_epoch90_fix_input.onnx",
    input_names=[
        "input",
    ],
    input_shapes=[
        [1, 3, 640, 640],
    ],
    output_names=[
        "dets",
        "labels",
    ],
    output_shapes=[
        [1, "num_dets", 5],
        [1, "num_dets"],
    ],
)
