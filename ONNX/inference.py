import torchvision.transforms as transforms
import onnxruntime
import cv2 as cv
import numpy as np
import torch
import onnx
import sys
import os

sys.path.append(os.getcwd())


class ONNXModel:
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  # 读取onnx模型权重
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self):
        """
        output_name = onnx_session.get_outputs()[0].name
        :return:
        """
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self):
        """
        input_name = onnx_session.get_inputs()[0].name
        :return:
        """
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        """
        image_numpy = image.transpose(2, 0, 1)
        image_numpy = image_numpy[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_numpy:
        :return:
        """
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        print(scores, boxes)
        return scores, boxes


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


weights = r'D:\Vortex\Project_2\sdyd_box\data\models\models\zhxjc\1.0.0\zhxjc_yolov5s_small.onnx'

to_tensor = transforms.ToTensor()

img = cv.imread(r'D:\Vortex\Project_4_RKNN\3399_pro_env_and_testCode\yolov5\bus.jpg')
img = cv.resize(img, (640, 640))
img = to_tensor(img)
img = img.unsqueeze_(0)

model = ONNXModel(weights)
inputs = {model.get_input_name()[0]: to_numpy(img)}
outputs = model.onnx_session.run(None, inputs)  # 只有最后的 output 才是最终的输出值
print(img.shape, outputs)
for a in outputs:
    print(a.shape)

# session = onnxruntime.InferenceSession(weights, None)
# raw_result = session.run([], {session.get_inputs()[0].name: to_numpy(img)})

# a = [np.zeros(shape=(1, 1000, 1)), np.zeros(shape=(1, 1000, 1))]
# b = np.concatenate(a, axis=1)
# print(b.shape)
