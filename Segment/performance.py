import os
import time
import math
import pathlib
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
from collections import Counter


def get_path(file_dir):
    """获取当前目录下，文件的绝对路径以及文件名称"""
    path_list = []
    name_list = []
    for path in pathlib.Path(file_dir).iterdir():
        path_list.append(str(path))
        name_list.append(path.name)
    path_list = sorted(path_list, key=lambda path_: int(pathlib.Path(path_).stem))
    name_list = sorted(name_list, key=lambda path_: int(pathlib.Path(path_).stem))
    return path_list, name_list


def prepro_image(img_path, img_resize, threshold=128):
    """将图像转换为二值图，并且resize"""
    image = cv.imread(img_path, 0)
    if len(image.shape) != 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, img_resize)
    _, bin_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    bin_image = np.array(bin_image / 255, dtype=np.int)
    return bin_image


def true_positive(pred, gt):
    assert pred.shape == gt.shape
    tp_bool = np.logical_and(pred, gt)
    tp_int = np.array(tp_bool, dtype=np.int)
    tp = np.sum(tp_int)
    return tp


def true_negative(pred, gt):
    assert pred.shape == gt.shape
    # 取反，背景为1、前景为0
    no_pred = np.array(np.logical_not(pred), dtype=np.int)
    no_gt = np.array(np.logical_not(gt), dtype=np.int)
    tn = true_positive(no_pred, no_gt)
    return tn


def false_positive(pred, gt):
    assert pred.shape == gt.shape
    # 取反，背景为1、前景为0
    no_gt = np.array(np.logical_not(gt), dtype=np.int)
    fp = true_positive(pred, no_gt)
    return fp


def false_negative(pred, gt):
    assert pred.shape == gt.shape
    # 取反，背景为1、前景为0
    no_pred = np.array(np.logical_not(pred), dtype=np.int)
    fn = true_positive(no_pred, gt)
    return fn


def calc_preformance(pred_path, gt_path, img_resize, threshold=128):
    """ 用于统计预测的分割图像与Ground Truth之间的差异，从而评价模型的性能, 但是目前只能计算灰度图像
    :param pred_path: 预测图像的路径
    :param gt_path: 真实mask路径
    :param img_resize: 将图像resize的尺寸
    :return: pix-accuracy、precision、recall、VOE、RVD、Dice、IOU评价指标
    """
    total_pix = reduce(lambda x, y: x * y, img_resize)
    pred_image = prepro_image(pred_path, img_resize, threshold)
    mask_image = prepro_image(gt_path, img_resize, threshold)

    # 计算 TP、FP、TN、FN
    start = time.time()
    tp = true_positive(pred_image, mask_image)
    tn = true_negative(pred_image, mask_image)
    fp = false_positive(pred_image, mask_image)
    fn = false_negative(pred_image, mask_image)

    # 计算评价指标
    accuracy = (tp + tn) / total_pix
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (fn + tp + tp + fp + 1e-10)
    voe = 1 - tp / (tp + fn + fp + 1e-10)
    rvd = (fp - fn) / (fn + tp + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    return tp, tn, fp, fn, accuracy, precision, recall, iou, dice, voe, rvd, specificity


def save_performace_to_csv(pred_dir, gt_dir, img_resize, csv_save_name, csv_save_path='', threshold=128):
    """ 保存到csv文件中
    :param gt_dir: 真实标签的目录
    :param pred_dir: 预测mask目录
    :param csv_save_name:
    :param img_resize: tuple 类型, 将图像resize后在计算 [height, width]
    :return:
    """
    gt_paths, gt_names = get_path(gt_dir)
    pred_paths, pred_names = get_path(pred_dir)

    record_pd = pd.DataFrame(columns=[
        'pred_name', 'gt_name', 'TP', 'FP', 'FN', 'TN',
        'accuracy', 'precision', 'recall', 'IOU', 'DICE', 'VOE', 'RVD', 'specificity',
    ])

    total_file_nums = len(gt_paths)
    for file_index in tqdm(range(total_file_nums), total=total_file_nums):
        TP, TN, FP, FN, accuracy, precision, recall, IOU, DICE, VOE, RVD, specificity = calc_preformance(
            pred_paths[file_index], gt_paths[file_index], img_resize, threshold)

        record_pd = record_pd.append({
            'pred_name': pred_names[file_index],
            'gt_name': gt_names[file_index],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'IOU': IOU,
            'DICE': DICE,
            'VOE': VOE,
            'RVD': RVD,
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
        }, ignore_index=True)

    record_pd.to_csv(
        os.path.join(csv_save_path, '{}.csv'.format(csv_save_name)), index=True, header=True)

    # 计算平均值
    m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec = analysis_performance(
        os.path.join(csv_save_path, '{}.csv'.format(csv_save_name)))
    analysis_pd = pd.DataFrame(columns=[
        'm_accu', 'm_prec', 'm_recall', 'm_iou', 'm_dice', 'm_voe', 'm_rvd', 'm_spec'
    ])
    analysis_pd = analysis_pd.append({
        'm_accu': m_accuracy, 'm_prec': m_precision, 'm_recall': m_recall, 'm_iou': m_iou,
        'm_dice': m_dice, 'm_voe': m_voe, 'm_rvd': m_rvd, 'm_spec': m_spec,
    }, ignore_index=True)
    analysis_pd.to_csv(
        os.path.join(csv_save_path, 'analysis_{}.csv'.format(csv_save_name)), index=True, header=True)
    return m_dice, m_iou, m_precision, m_recall


def analysis_performance(csv_file_path):
    """  统计csv文件中各个performance的平均值 """
    data_frame = pd.read_csv(csv_file_path, header=None)

    m_accuracy = np.mean(np.array(data_frame.loc[1:, 7], dtype=np.float32))
    m_precision = np.mean(np.array(data_frame.loc[1:, 8], dtype=np.float32))
    m_recall = np.mean(np.array(data_frame.loc[1:, 9], dtype=np.float32))
    m_iou = np.mean(np.array(data_frame.loc[1:, 10], dtype=np.float32))
    m_dice = np.mean(np.array(data_frame.loc[1:, 11], dtype=np.float32))
    m_voe = np.mean(np.array(data_frame.loc[1:, 12], dtype=np.float32))
    m_rvd = np.mean(np.array(data_frame.loc[1:, 13], dtype=np.float32))
    m_spec = np.mean(np.array(data_frame.loc[1:, 14], dtype=np.float32))
    print(
        ' accuracy: {},\n precision: {},\n recall: {},\n iou: {},\n dice: {},\n voe: {},\n rvd: {},\n spec: {}.\n'.format(
            m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec))
    return m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


if __name__ == '__main__':
    # pred_dir_ = r'E:\tookit_backup\毕业论文\程序\DWNET\invalid_pred_crop\epoch_1'
    # gt_dir_ = r'E:\tookit_backup\毕业论文\程序\data\invalid_mask'
    # save_performace_to_csv(pred_dir_, gt_dir_,
    #                        (256, 256),
    #                        'test', r'E:\tookit_backup\毕业论文\程序\DWNET\invalid_pred_crop')
    img_path_ = r'D:\Users\YingYing\Desktop\data\unet_bn_predict_mask\pred_mask\007582.jpg'
    bin_image = prepro_image(img_path_, (256, 256))
    print(Counter(np.reshape(bin_image, (-1,))))
