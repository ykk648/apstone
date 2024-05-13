import onnx
import onnx.helper

onnx_p = ''

# 加载现有的 ONNX 模型
original_model = onnx.load(onnx_p)

# 创建一个新的 ONNX 模型，复制原始模型的图结构
new_model = onnx.ModelProto()
new_model.graph.CopyFrom(original_model.graph)

# 添加描述信息到新模型的文档字符串中
model_doc_string = "test"
new_model.doc_string = model_doc_string

# 保存新模型到文件
onnx.save(new_model, "updated_model.onnx")


# 加载 ONNX 模型
model = onnx.load("updated_model.onnx")

# 读取模型的元数据
if model.metadata_props:
    print("\nModel Metadata:")
    for metadata in model.metadata_props:
        print(f"{metadata.key}: {metadata.value}")
else:
    print("\nModel does not contain any metadata.")