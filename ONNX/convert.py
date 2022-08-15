import torch
import onnx
import sys

sys.path.insert(0, './Classifier/torch')

"""将模型转换为 onnx"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load(r"C:\Users\yuhe\Desktop\model_weights\left_1_8_1.pt", map_location='cpu')  # pytorch模型加载
batch_size = 1  # 批处理大小
input_shape = (3, 224, 224)  # 输入数据,改成自己的输入shape

# set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)  # 生成张量
# x = x.to(device)
export_onnx_file = "efficient_lite2_left_1_8_1.onnx"  # 目的ONNX文件名
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=11,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  # dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                  #               "output": {0: "batch_size"}}
                  )
