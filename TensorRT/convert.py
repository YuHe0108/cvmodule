"""将 onnx -> tensorRT"""
import os
import cv2
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_file_path, engine_file_path, input_shape):
    """将 onnx 权重转换为 trt 文件"""
    # 基于 INetworkDefinition 构建 ICudaEngine
    builder = trt.Builder(TRT_LOGGER)
    # 基于 INetworkDefinition 和 IBuilderConfig 构建 engine
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 构建 builder 的配置对象
    config = builder.create_builder_config()
    # 构建 ONNX 解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # 构建 TensorRT 运行时
    runtime = trt.Runtime(TRT_LOGGER)
    # 参数设置
    config.max_workspace_size = 1 << 28  # 256MiB
    builder.max_batch_size = 1
    # 解析 onnx 模型
    if not os.path.exists(onnx_file_path):
        print(
            f"[INFO] ONNX file {onnx_file_path} not found.")
    print(f"[INFO] Loading ONNX file from {onnx_file_path}.")
    with open(onnx_file_path, "rb") as model:
        print("[INFO] Beginning ONNX file parsing.")
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            # 根据 yolov3.onnx，reshape 输入数据的形状
    network.get_input(0).shape = [1, 3, input_shape[0], input_shape[1]]
    print("[INFO] Completed parsing of ONNX file.")
    print(f"[INFO] Building an engine from {onnx_file_path}.")
    # 序列化模型
    plan = builder.build_serialized_network(network, config)
    # 反序列化
    engine = runtime.deserialize_cuda_engine(plan)
    print("[INFO] Completed creating engine.")
    # 写入文件
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    return engine


if __name__ == "__main__":
    build_engine("", "", "")
