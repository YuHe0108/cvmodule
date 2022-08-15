"""使用转换后的模型进行测试"""

import cv2
import numpy as np


def load_rknn_model(model_file):
    """通过 .rknn 文件加载模型"""
    from rknn.api import RKNN
    # 模型加载之前需要进行解密处理
    rknn = RKNN()
    ret = rknn.load_rknn(model_file)
    if ret != 0:
        print('load rknn model failed')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
    return rknn


def img_processing(img, new_shape=(384, 384), color=(0, 0, 0)):
    """
    resize 图像尺寸, 保证了图像的长宽比例不变
    """
    shape = img.shape[:2]
    if tuple(shape) == new_shape:
        return img
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


image = cv2.imread('left.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = img_processing(image, new_shape=(224, 224))  # 对图像尺寸进行调整

model = load_rknn_model('left_obj_efficient.rknn')
outputs = model.inference(inputs=[image])[0]
outputs = np.exp(outputs[0])
print(outputs)
