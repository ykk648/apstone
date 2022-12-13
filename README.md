### Introduction

Base stone of AI_power, maintain all inference of AI_Power models.

#### Wrapper

- Supply different model infer wrapper, including ONNX/TensorRT/Torch JIT;
- Support onnx different Execution Providers (EP) , including cpu/gpu/trt/trt16/int8;
- High level mmlab model (converted) infer wrapper, including MMPose/MMDet;

#### Model Convert

- torch2jit torch2onnx etc.
- detectron2 to onnx
- modelscope to onnx
- onnx2simple2trt
- tf2pb2onnx

#### Model Tools

- torch model edit
- onnx model shape/speed test (different EP)
- common scripts from onnxruntime

### Usage

#### onnx model speed test
```python
from apstone import ONNXModel

onnx_p = 'pretrain_models/sr_lib/realesr-general-x4v3-dynamic.onnx'
input_dynamic_shape = (1, 3, 96, 72)  # None
# cpu gpu trt trt16 int8
ONNXModel(onnx_p, provider='cpu', debug=True, input_dynamic_shape=input_dynamic_shape).speed_test()
```

### Install

```sh
pip install apstone
```

#### Envs

| Execution Providers | Needs                                                       |
| ------------------- | ----------------------------------------------------------- |
| cpu                 | pip install onnxruntime                                     |
| gpu                 | pip install onnxruntime-gpu                                 |
| trt/trt16/int8      | onnxruntime-gpu compiled with tensorrt EP                   |
| TensorRT            | pip install tensorrt pycuda                                 |
| torch JIT           | install [pytorch](https://pytorch.org/get-started/locally/) |

