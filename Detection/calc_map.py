from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv
import pathlib
import torch
import json
import tqdm
import sys
import os

sys.path.insert(0, "..\\Detection\\yolov5")
sys.path.insert(0, "..\\")

import tools
from Detection.yolov5.utils.datasets import letterbox
from Detection.yolov5.utils.general import non_max_suppression, scale_coords

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
        self.input_shape = img_shape
        self.weight_path = weight_path

        self.conv_detect_results = False  # 已经存在的预测结果是否覆盖

        self.root_dir = 'run_map_results'  # 将 预测结果、计算 map 产生的结果统一存放的路径
        self.pred_dir = None  # 预测结果 txt 文件的保存位置
        self.label_txt_dir = None  # txt 标签抓换后的存放路径
        self.save_calc_results_dir = None  # 计算 map 保存结果的路径
        self.label_json_dir = os.path.join(self.root_dir, 'label_json')  # 将 label 标签重新保存为 json 格式 的路径

        self.show_animation = False  # 显示结果
        self.draw_plot = False  # 绘制结果图

        self.min_score_thresh = 0.5  # 置信度大于该值时，认为是目标
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        self.idx2label = self.read_label()  # 标签有哪些
        self.label2idx = {label: i for i, label in self.idx2label.items()}
        self.ignore_labels = set()  # 哪些类不参与 map 计算，自行添加

    def calculate(self):
        self.xml2txt()  # 首先将 xml 文件转换为 txt
        self.pred()  # 对每张图预测推理，并将推理结果保存至本地 txt 文档
        self.calc_map()  # 根据预测结果和推理结果计算 map
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
        img_dir = self.img_dir if self.img_dir is not None else self.data_dir  # 存放 invalid 图像的路径
        self.pred_dir = os.path.join(self.root_dir, 'predict_txt')  # 预测结果的存放路径
        tools.check_path(self.pred_dir)

        print('开始推理图像....')
        for img_file in tqdm.tqdm(pathlib.Path(img_dir).iterdir()):
            if img_file.suffix not in ['.jpg', 'jpeg', 'JPG', 'JPEG', 'png']:
                continue
            img_name = pathlib.Path(img_file).stem  # 预测图像的名字
            # 允许覆盖，并且预测的 txt 结果已经存在，则不在重复计算
            if not self.conv_detect_results and os.path.exists(os.path.join(self.pred_dir, "{}.txt".format(img_name))):
                continue
            inputs, ori_img = self.read_img(str(img_file))  # 读取图像
            boxes, scores, classes = self.inference(inputs, ori_img.shape)  # 通过模型推理出的结果
            self.save_pred_txt(img_name, ori_img, boxes, scores, classes)  # 将预测的结果保存至本地
        print('推理阶段完成！')
        return

    def calc_map(self):
        # self.label_txt_dir
        # self.pred_dir
        self.save_calc_results_dir = os.path.join(self.root_dir, 'calc_results')  # 结果保存的路径
        plot_dir = os.path.join(self.save_calc_results_dir, 'classes')  # 保存 plot 的结果
        animation = os.path.join(self.save_calc_results_dir, 'images', 'detections_one_by_one')
        tools.check_path([self.label_json_dir, self.save_calc_results_dir, plot_dir, animation])

        gt_counter_per_class = defaultdict(int)  # 记录每个类在总数据集中的数量
        counter_images_per_class = defaultdict(int)  # 记录每个类出现在多少张图像中
        miss_detector_cnt = 0  # 缺少多少预测的 txt 文件

        gt_json_files = []
        # 遍历label标签
        for label_txt_file in pathlib.Path(self.label_txt_dir).iterdir():
            if pathlib.Path(label_txt_file).suffix != '.txt':
                continue

            label_txt_file_name = pathlib.Path(label_txt_file).stem
            pred_txt_file_path = os.path.join(self.pred_dir, f'{label_txt_file_name}.txt')  # 预测的txt路径
            if not os.path.exists(pred_txt_file_path):
                print(f'预测文件不存在：{label_txt_file_name}')
                miss_detector_cnt += 1
                continue

            is_difficult = False
            bounding_boxes = []  # 真实标注框的信息
            already_seen_classes = set()  # 当前图像有相同类出现两个以上时，只算一次
            lines_list = tools.file_lines_to_list(
                os.path.join(self.label_txt_dir, label_txt_file_name))  # 读取txt文档，并返回列表形式
            class_name = left = top = right = bottom = _difficult = None
            for line in lines_list:
                # 读取 label txt 信息
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                except ValueError:
                    print("预测 txt 文档格式不正确！")

                if class_name in self.ignore_labels:  # 是否放弃计算某些类的 map
                    continue

                bbox = left + " " + top + " " + right + " " + bottom  # 目标预测框信息
                if is_difficult:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

                    gt_counter_per_class[class_name] += 1  # 记录每个类的数量

                    if class_name not in already_seen_classes:  # 记录每个类在出现在多少张图像中
                        already_seen_classes.add(class_name)
                        counter_images_per_class[class_name] += 1

            # 将标签重新保存为 json 格式
            label_json_file = os.path.join(self.label_json_dir, f"{label_txt_file_name}_ground_truth.json")
            gt_json_files.append(label_json_file)
            with open(label_json_file, 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())  # 数据集中存在哪些类
        gt_classes = sorted(gt_classes)  # 排序
        n_classes = len(gt_classes)  # 数据集中一共出现类的种类数量

        # 将 预测的 txt 同样转换为 json 格式
        for class_index, class_name in enumerate(gt_classes):  # 按照类别逐个计算
            bounding_boxes = []
            pred_class_name = confidence = left = top = right = bottom = None
            for pred_txt_file in pathlib.Path(self.pred_dir).iterdir():
                pred_txt_file_name = pathlib.Path(pred_txt_file).stem
                label_txt_file = os.path.join(self.label_txt_dir, f'{pred_txt_file_name}.txt')
                if class_index == 0:
                    if not os.path.exists(label_txt_file):
                        print('label txt file not exist', label_txt_file)

                lines = tools.file_lines_to_list(pred_txt_file)
                for line in lines:
                    try:
                        pred_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        print(f'读取模型预测的txt文件错误：{pred_txt_file_name}.')

                    if pred_class_name == class_name:
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_name": pred_txt_file_name, "bbox": bbox})

            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # 按照置信度从大到小排序, 并保存为json文件
            with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        print(f'计算完成，一共缺少{miss_detector_cnt}个文件没有参与计算。')
        return

    def get_model(self):
        """ 加载模型 """
        model = torch.load(self.weight_path)['model'].eval().float().to(self.device)  # 加载模型
        return model

    def read_img(self, img_path):
        ori_img = cv.imread(img_path)  # 原始图
        img = letterbox(ori_img, self.input_shape, stride=64, auto=False)[0]  # 缩放后的图
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img, ori_img

    def inference(self, inputs, ori_img_shape):
        # 通过模型得到推理结果
        pred = self.model(torch.tensor(inputs).to(self.device))
        pred = pred[0][None, ...]
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=100)
        gn = torch.tensor(ori_img_shape)[[1, 0, 1, 0]]  # [3, 1, 3, 1]

        boxes = []
        scores = []
        classes = []
        for i, det in enumerate(pred):
            # 坐标: scale_coords:
            det[:, :4] = scale_coords(inputs.shape[2:], det[:, :4], ori_img_shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1y1x2y2 = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                boxes.append([x1y1x2y2[1], x1y1x2y2[0], x1y1x2y2[3], x1y1x2y2[2]])
                scores.append(float(conf))
                classes.append(int(cls))

        boxes = np.expand_dims(boxes, 0)
        scores = np.expand_dims(scores, 0)
        classes = np.expand_dims(classes, 0)
        return boxes, scores, classes

    def save_pred_txt(self, img_name, ori_img, boxes, scores, classes):
        height, width = ori_img.shape[:2]
        result_list = []
        for idx, class_id in enumerate(classes[0]):
            conf_score = round(float(scores[0, idx]), 3)  # 置信度保存小数点后三位
            if conf_score > self.min_score_thresh:
                bbox = boxes[0, idx]
                ymin, xmin = int(bbox[0] * height), int(bbox[1] * width)
                ymax, xmax = int(bbox[2] * height), int(bbox[3] * width)
                msg = "{} {} {} {} {} {}\n".format(
                    self.idx2label[int(class_id)], conf_score, xmin, ymin, xmax, ymax)  # 类别、置信度和坐标
                result_list.append(msg)
        with open(os.path.join(self.pred_dir, "{}.txt".format(img_name)), "w") as f:  # 以图像名字为txt名字进行保存
            f.writelines(result_list)  # class_id, conf_score, xmin, ymin, xmax, ymax
        return

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
