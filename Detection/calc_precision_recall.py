import os
import cv2
import sys
import pathlib
from collections import defaultdict

sys.path.insert(0, '.')

from tools import check_path
from utils.caculate import calc_iou
from utils.img_utils import draw_img


def calc(pred_path, target_path, img_path, save_dir):
    """计算未检出、误报"""
    save_not_check_out = os.path.join(save_dir, 'res/not_check_out')
    save_error_detect = os.path.join(save_dir, 'res/error_detect')
    check_path([save_not_check_out, save_error_detect])
    with open(pred_path, 'r') as file:
        lines = file.readlines()
        pred_res = [line.strip() for line in lines]

    with open(target_path, 'r') as file:
        lines = file.readlines()
        targ_res = [line.strip() for line in lines]

    # 统计
    iou_count = {i: [-1, -1] for i in range(len(targ_res))}
    error_count = []  # 错误的标记出来
    predict_box = []
    target_box = []
    for i, p_info in enumerate(pred_res):
        p_cls, _, p_x1, p_y1, p_x2, p_y2 = [eval(x) for x in p_info.split(' ')]
        predict_box.append([p_x1, p_y1, p_x2, p_y2, p_cls])
        max_iou = -float('inf')
        correspond_cls = -1
        for j, t_info in enumerate(targ_res):
            t_cls, t_x1, t_y1, t_x2, t_y2 = [int(x) for x in t_info.split(' ')]
            target_box.append([t_x1, t_y1, t_x2, t_y2, t_cls])
            cur_iou = calc_iou([t_x1, t_y1, t_x2, t_y2], [p_x1, p_y1, p_x2, p_y2])
            if t_cls != p_cls:  # 保证类别一致
                continue
            if cur_iou > max_iou:
                max_iou = cur_iou
                correspond_cls = j  # 最大的 iou 对应匹配真值的索引

        if correspond_cls == -1:  # 类别不同
            error_count.append(1)
        elif iou_count[correspond_cls][0] == -1:
            iou_count[correspond_cls] = [max_iou, i]
        else:
            error_count.append(iou_count[correspond_cls][1])

    # 计算差异
    img = cv2.imread(img_path)
    img_name = pathlib.Path(img_path).name
    print(iou_count, img_name, error_count)
    for k, v in iou_count.items():  # 根据每个目标框的
        iou_score = v[0]
        if iou_score < 0.6:  # 小于0.6, 没有匹配成功, 漏检
            img = draw_img(img, predict_box, (0, 0, 255), 'P')
            img = draw_img(img, target_box, (255, 0, 0), 'R')
            cv2.imwrite(os.path.join(save_not_check_out, img_name), img)

    # 误检
    if len(error_count) > 0:
        # for k, v in iou_count.items():  # 根据每个真实目标框的IOU值
        #     iou_score = v[0]
        #     if iou_score < 0.6:
        img = draw_img(img, predict_box, (0, 0, 255), 'P')
        img = draw_img(img, target_box, (255, 0, 0), 'R')
        cv2.imwrite(os.path.join(save_error_detect, img_name), img)
    return


def run(pred_dir, target_dir, img_dir, save_dir):
    img_suffix = ['.jpg', '.jpeg', '.png', 'JPG']
    for path in pathlib.Path(target_dir).iterdir():  # 寻找标签txt目录下的文件
        if path.suffix == '.txt':
            pred_txt_path = os.path.join(pred_dir, path.name)
            stem = path.stem
            img_path = None
            for suffix in img_suffix:
                img_path = os.path.join(img_dir, f'{stem}{suffix}')
                if os.path.exists(img_path):
                    break
            if img_path is not None:
                calc(pred_txt_path, str(path), img_path, save_dir)
    return


if __name__ == '__main__':
    run(r"C:\Users\yuhe\Desktop\calc\predict_txt_0901",
        r"C:\Users\yuhe\Desktop\image_test",
        r'C:\Users\yuhe\Desktop\image_test',
        save_dir=r'C:\Users\yuhe\Desktop\draw')
