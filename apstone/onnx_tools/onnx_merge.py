# -- coding: utf-8 --
# @Time : 2023/2/21
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
"""
ref https://github.com/PINTO0309/snc4onnx

pip install -U onnx \
&& pip install -U onnx-simplifier \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U snc4onnx
"""

# from snc4onnx import combine
#
# combined_graph = combine(
#     srcop_destop=[
#         ['output', 'flow_init']
#     ],
#     op_prefixes_after_merging=[
#         'init',
#         'next',
#     ],
#     input_onnx_file_paths=[
#         'crestereo_init_iter2_120x160.onnx',
#         'crestereo_next_iter2_240x320.onnx',
#     ],
#     non_verbose=True,
# )

"""
ref https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#onnx-compose
"""
import onnx

model1 = onnx.load("path/to/model1.onnx")
new_model1 = onnx.compose.add_prefix(model1, prefix="m1/")

model2 = onnx.load("path/to/model2.onnx")
new_model2 = onnx.compose.add_prefix(model2, prefix="m2/")

combined_model = onnx.compose.merge_models(
    new_model1, new_model2,
    io_map=[
        ("m1/C", "m2/X"),
        ("m1/D", "m2/Y")
    ]
)
