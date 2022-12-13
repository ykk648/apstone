from cv2box import try_import
torch = try_import('torch', 'torch2any need torch !')


def torch2jit(model_, input_):
    traced_gpu = torch.jit.trace(model_, input_)
    torch.jit.save(traced_gpu, "gpu.tjm")


def torch2onnx(model_, input_, output_name="./test.onnx"):
    input_names = ["input_1"]
    output_names = ["output_1"]
    opset_version = 13
    dynamic_axes = None
    # dynamic_axes = {'input_1': [0, 2, 3], 'output_1': [0, 1]}
    torch.onnx.export(model_, input_, output_name, verbose=True, opset_version=opset_version,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, do_constant_folding=True)
    raise 'convert done !'
