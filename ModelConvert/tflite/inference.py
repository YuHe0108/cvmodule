import torchvision.transforms as transforms
import tensorflow as tf
import numpy as np
import cv2 as cv
import torch

from Detection.yolov5.utils.general import non_max_suppression


def run():
    return


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


interpreter = tf.lite.Interpreter(model_path=r"D:\Vortex\SELF\cvmodule\Detection\yolov5\weights\yolov5s-fp16.tflite")
interpreter.allocate_tensors()

# 模型输入和输出细节
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

to_tensor = transforms.ToTensor()

img = cv.imread(r'D:\Vortex\Project_4_RKNN\3399_pro_env_and_testCode\yolov5\bus.jpg')
img = cv.resize(img, (640, 640))
img = to_tensor(img)
img = img.permute(1, 2, 0)
img = img.unsqueeze_(0)
imgs = torch.concat([img, img])

imgs = to_numpy(imgs)

# 模型预测
interpreter.set_tensor(input_details[0]['index'], imgs)  # 传入的数据必须为ndarray类型
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
output = torch.tensor(output)
pred = non_max_suppression(output, 0.25, 0.45, None, False, max_det=100)
for a in pred:
    a = to_numpy(a)
    coord = a[:, :4]
    coord = coord * 640  # 目标的尺寸
    print(coord)
