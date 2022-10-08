import os
import cv2
import sys
import tqdm
import pathlib
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, '.')

from tools import check_path
from utils.caculate import calc_iou
from utils.img_utils import draw_img

INT_TO_LABEL = None  # 数字转换为字符串
USEFUL_CLASS_IDX = {0, 1, 2, 3}  # 需要参与计算的类别


def calc(pred_path, target_path, img_path, save_dir, iou_threshold=0.5):
    """计算未检出、误报
    pred_path:      预测的 txt 文件
    target_path:    gt 的 txt 文件
    img_path:       图像存放的位置
    iou_threshold:  预测和真值的iou超过 0.5 认为匹配成功
    """
    global INT_TO_LABEL, USEFUL_CLASS_IDX

    save_negative_samples_path = os.path.join(save_dir, 'NegativeSamples')
    check_path([save_negative_samples_path])
    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        print(f'not exist: {pred_path} or {target_path}')
        return
    with open(pred_path, 'r') as file:
        lines = file.readlines()
        pred_res = [line.strip() for line in lines]
    with open(target_path, 'r') as file:
        lines = file.readlines()
        targ_res = [line.strip() for line in lines]

    # 统计
    error_count = []  # 错误的标记出来
    predict_box = []
    target_box = []
    seen = set()
    iou_count = {i: [-1, -1] for i in range(len(targ_res))}
    for i, p_info in enumerate(pred_res):
        p_cls, _, p_x1, p_y1, p_x2, p_y2 = [eval(x) for x in p_info.split(' ')]
        if int(p_cls) not in USEFUL_CLASS_IDX:
            continue
        predict_box.append([p_x1, p_y1, p_x2, p_y2, p_cls])
        max_iou = -float('inf')
        correspond_cls = -1

        for j, t_info in enumerate(targ_res):
            t_cls, t_x1, t_y1, t_x2, t_y2 = [int(x) for x in t_info.split(' ')]
            if int(t_cls) not in USEFUL_CLASS_IDX:
                continue

            if j not in seen:  # 避免重复添加
                seen.add(j)
                target_box.append([t_x1, t_y1, t_x2, t_y2, t_cls])
            cur_iou = calc_iou([t_x1, t_y1, t_x2, t_y2], [p_x1, p_y1, p_x2, p_y2])
            if t_cls != p_cls:  # 保证类别一致
                continue
            if cur_iou > max_iou and cur_iou > iou_threshold:
                max_iou = cur_iou
                correspond_cls = j  # 最大的 iou 对应匹配真值的索引

        if correspond_cls == -1:  # 当前的预测框，没有与任何真值框匹配成功
            error_count.append(i)
        elif iou_count[correspond_cls][0] == -1:  # 匹配成功
            iou_count[correspond_cls] = [max_iou, i]
        else:
            error_count.append(iou_count[correspond_cls][1])  # 之前匹配成功的预测标签

    # 计算差异
    can_write = False  # 是否存在预测问题
    img = cv2.cvtColor(plt.imread(img_path), cv2.COLOR_RGB2BGR)
    img_name = pathlib.Path(img_path).name
    for i in range(len(predict_box)):
        if i in error_count:  # 当前的预测框：1：没有与任何真值框匹配成功，2：当前框不是最佳匹配框
            can_write = True
            img = draw_img(img, [predict_box[i]], (255, 0, 0), INT_TO_LABEL, 'P')
    if len(iou_count) > 0:
        for i in range(len(target_box)):
            max_iou, _ = iou_count[i]
            if max_iou == -1:
                can_write = True
                img = draw_img(img, [target_box[i]], (0, 0, 255), INT_TO_LABEL, 'R')
    if can_write:
        cv2.imwrite(os.path.join(save_negative_samples_path, img_name), img)
    return


def run(pred_dir, target_dir, img_dir, save_dir, label_txt):
    """
    label_txt: 标记在图中时，将数字转换为字符串
    """
    # 通过 label_txt 文件读取类别标签
    global INT_TO_LABEL
    assert os.path.exists(label_txt)
    with open(label_txt, 'r') as file:
        labels = file.readlines()
    INT_TO_LABEL = {i: label.strip() for i, label in enumerate(labels)}

    img_suffix = ['.jpg', '.jpeg', '.png', '.JPG']
    for path in tqdm.tqdm(pathlib.Path(target_dir).iterdir()):  # 寻找标签txt目录下的文件
        if path.suffix == '.txt':
            pred_txt_path = os.path.join(pred_dir, path.name)
            stem = path.stem
            img_path = None
            for suffix in img_suffix:
                img_path = os.path.join(img_dir, f'{stem}{suffix}')
                if os.path.exists(img_path):
                    break
            try:
                if img_path is not None:
                    calc(pred_txt_path, str(path), img_path, save_dir)
            except Exception as e:
                print(img_path, stem, path, e)
    return


if __name__ == '__main__':
    target_file = "3"
    run(fr"C:\Users\yuhe\Desktop\valid_data\predict\ori_pred\0902-1\{target_file}",
        fr"C:\Users\yuhe\Desktop\valid_data\{target_file}",
        fr'C:\Users\yuhe\Desktop\valid_data\{target_file}',
        save_dir=r'C:\Users\yuhe\Desktop\draw',
        label_txt=r'C:\Users\yuhe\Desktop\valid_data\label.txt')
