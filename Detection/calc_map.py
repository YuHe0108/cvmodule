import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv
import pathlib
import torch
import sys
import os

sys.path.insert(0, "..\\Detection\\yolov5")
sys.path.insert(0, "..\\")

import tools
from Detection.yolov5.utils.datasets import letterbox
from Detection.yolov5.utils.general import non_max_suppression

"""计算模型的 map """


class CalcMAP:
    def __init__(self, weight_path, label_file, img_shape, data_dir=None, img_dir=None, xml_dir=None):
        """
        img_shape:      测试 resize 图像尺寸
        label_file:     标签文件，txt 格式即可
        data_dir:       图像和 xml 文件放在一起，则只提供当前路径即可
        img_dir:        验证集图像的路径
        xml_dir:        对应图像 xml 标签的路径
        """
        assert data_dir is not None or (img_dir is not None and xml_dir is not None)
        if data_dir is not None:
            assert os.path.exists(data_dir)
        if img_dir is not None and xml_dir is not None:
            assert os.path.exists(img_dir) and os.path.exists(xml_dir)

        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.data_dir = data_dir
        self.label_file = label_file
        self.img_shape = img_shape
        self.weight_path = weight_path
        self.label_txt_dir = None  # txt 标签抓换后的存放路径
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        self.idx2label = self.read_label()  # 标签有哪些
        self.label2idx = {label: i for i, label in self.idx2label.items()}

    def get_model(self):
        """ 加载模型 """
        model = torch.load(self.weight_path)['model'].eval().float().to(self.device)  # 加载模型
        return model

    def calculate(self):
        self.xml2txt()  # 首先将 xml 文件转换为 txt
        self.pred()
        return

    def xml2txt(self):
        """将 xml 文件转换成 txt 文件, 并保存在与 xml 相同的路径下"""
        xml_dir = self.xml_dir if self.xml_dir is not None else self.data_dir
        self.label_txt_dir = xml_dir
        for xml_file in pathlib.Path(xml_dir).iterdir():
            file_name = pathlib.Path(xml_file).stem
            save_txt_path = os.path.join(xml_dir, "{}.txt".format(file_name))  # 需要保存的txt文件
            if xml_file.suffix != '.xml' or os.path.exists(save_txt_path):
                continue

            # 读取 xml 文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            result_list = []
            for member in root.findall('object'):
                class_name = member[0].text
                idx = self.label2idx.get(class_name, -1)  # 当前不存在于标签中
                if idx < 0:
                    continue
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                msg = "{} {} {} {} {} \n".format(class_name, xmin, ymin, xmax, ymax)
                result_list.append(msg)

            with open(save_txt_path, "w") as f:
                # class_id, conf_score, xmin, ymin, xmax, ymax
                f.writelines(result_list)
        print('xml convert to txt done !')
        return

    def pred(self):
        img_dir = self.img_dir if self.img_dir is not None else self.data_dir
        save_dir = os.path.join(pathlib.Path(img_dir).parent, 'predict_txt')  # 预测结果的存放路径
        tools.check_path(save_dir)
        for img_file in pathlib.Path(img_dir).iterdir():
            if img_file.suffix not in ['.jpg', 'jpeg', 'JPG', 'JPEG', 'png']:
                continue
            inputs, (ori_height, ori_width) = self.read_img(str(img_file))  # 读取图像
            outputs = self.inference(inputs, (ori_height, ori_width))

        return

    def inference(self, inputs, ori_img_shape):
        # 通过模型进行推理
        print(inputs.shape)
        pred = self.model(torch.tensor(inputs).to(self.device))
        pred = pred[0][None, ...]
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=100)
        gn = torch.tensor(inputs.shape)[[1, 0, 1, 0]]
        print(gn.shape)

        boxes = []
        scores = []
        classes = []
        copied = inputs.copy()
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1y1x2y2 = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()

                boxes.append([x1y1x2y2[1], x1y1x2y2[0], x1y1x2y2[3], x1y1x2y2[2]])
                scores.append(float(conf))
                classes.append(int(cls) + 1)

        boxes = np.expand_dims(boxes, 0)
        scores = np.expand_dims(scores, 0)
        classes = np.expand_dims(classes, 0)
        return

    def read_img(self, img_path):
        img = cv.imread(img_path)
        height, width = img.shape[:2]
        img = letterbox(img, self.img_shape, stride=64, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img, (height, width)

    def read_label(self):
        with open(self.label_file, 'r') as file:
            labels = file.readlines()
        idx2label = {i: label.strip() for i, label in enumerate(labels)}
        return idx2label


if __name__ == '__main__':
    WEIGHT_PATH = r'D:\Vortex\Project_7_huzhou\waste_trash_device1280_v1.24.pt'  # 模型权重路径
    DATA_DIR = r'D:\Vortex\Project_7_huzhou\invalid'  # 数据集的路径
    LABEL_FILE = r'D:\Vortex\SELF\cvmodule\Detection\yolov5\data\label.txt'
    calc_map = CalcMAP(WEIGHT_PATH, LABEL_FILE, (1280, 1280), DATA_DIR)
    calc_map.calculate()
