import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
import cv2
import os

from TensorRT.utils import processing_image, allocate_buffers, do_inference, get_engine


def run(trt_path, img_path):
    # 获取推理引擎
    engine = get_engine(trt_path)
    # 创建推理上下文
    context = engine.create_execution_context()
    # 分配内存
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 为图像分配主机内存
    image = processing_image(cv2.imread(img_path), (224, 224))
    inputs[0].host = image
    # 推理并获得输出
    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
    # 由于 trt_outputs 为展开的张量，这里将其 reshape
    print(trt_outputs)
    return


if __name__ == '__main__':
    run('', '')
