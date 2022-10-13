import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from utils.img_utils import letterbox

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def processing_image(image, img_shape):
    image = cv2.resize(image, img_shape)
    image = image.astype(np.float32)
    # 归一化
    image /= 255.0
    # (w,h,c) -> (c,h,w)
    image = np.transpose(image, [2, 0, 1])
    # (c,h,w) -> (n,c,h,w)
    image = np.expand_dims(image, axis=0)
    image = np.array(image, dtype=np.float32, order="C")
    return image


def get_engine(onnx_file_path, engine_file_path=""):
    # 如果不指定 engine_file_path 则通过 build_engine 生成 engine 文件
    def build_engine():
        # 基于 INetworkDefinition 构建 ICudaEngine
        builder = trt.Builder(TRT_LOGGER)
        # 基于 INetworkDefinition 和 IBuilderConfig 构建 engine
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
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
        network.get_input(0).shape = [1, 3, 608, 608]
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

    if os.path.exists(engine_file_path):
        print(f"[INFO] Reading engine from {engine_file_path}.")
        with open(engine_file_path, "rb") as f:
            with trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(
            engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 分配主机内存和设备内存
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 绑定设备内存
        bindings.append(int(device_mem))
        # 输入输出绑定
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # 将输入数据从主机拷贝到设备
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 将输出数据从设备拷贝到主机
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 同步流
    stream.synchronize()
    # 仅返回主机上的输出
    return [out.host for out in outputs]
