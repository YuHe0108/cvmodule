import tensorflow as tf
import numpy as np
import torch
import math

epsilon = 1e-5


def IoU(box1, box2, wh=False):
    if wh:
        xmin1, ymin1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        xmax1, ymax1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        xmin2, ymin2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        xmax2, ymax2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

    # 计算交集部分尺寸
    W = min(xmax1, xmax2) - max(xmin1, xmin2)
    H = min(ymax1, ymax2) - max(ymin1, ymin2)

    # 计算两个矩形框面积
    SA = (xmax1 - xmin1) * (ymax1 - ymin1)
    SB = (xmax2 - xmin2) * (ymax2 - ymin2)

    cross = max(0, W) * max(0, H)  # 计算交集面积
    iou = float(cross) / (SA + SB - cross)

    return iou


def GIoU(box1, box2, wh=False):
    if wh:
        xmin1, ymin1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        xmax1, ymax1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        xmin2, ymin2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        xmax2, ymax2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

    iou = IoU(box1, box2, wh)
    SC = (max(xmax1, xmax2) - min(xmin1, xmin2)) * (max(ymax1, ymax2) - min(ymin1, ymin2))

    # 计算交集部分尺寸
    W = min(xmax1, xmax2) - max(xmin1, xmin2)
    H = min(ymax1, ymax2) - max(ymin1, ymin2)

    # 计算两个矩形框面积
    SA = (xmax1 - xmin1) * (ymax1 - ymin1)
    SB = (xmax2 - xmin2) * (ymax2 - ymin2)

    cross = max(0, W) * max(0, H)  # 计算交集面积

    add_area = SA + SB - cross  # 两矩形并集的面积

    end_area = (SC - add_area) / SC  # 闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    return giou


def DIoU(box1, box2, wh=False):
    if wh:
        inter_diag = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
        xmin1, ymin1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        xmax1, ymax1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        xmin2, ymin2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        xmax2, ymax2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        center_x1 = (xmax1 + xmin1) / 2
        center_y1 = (ymax1 + ymin1) / 2
        center_x2 = (xmax2 + xmin2) / 2
        center_y2 = (ymax2 + ymin2) / 2
        inter_diag = (center_x1 - center_x2) / 2 ** 2 + (center_y1 - center_y2) ** 2

    iou = IoU(box1, box2, wh)
    enclose1 = max(max(xmax1, xmax2) - min(xmin1, xmin2), 0.0)
    enclose2 = max(max(ymax1, ymax2) - min(ymin1, ymin2), 0.0)
    outer_diag = (enclose1 ** 2) + (enclose2 ** 2)
    diou = iou - 1.0 * inter_diag / outer_diag
    return diou


def CIoU(box1, box2, wh=False, normaled=False):
    if wh:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        inter_diag = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
        xmin1, ymin1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        xmax1, ymax1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        xmin2, ymin2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        xmax2, ymax2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        w1, h1 = xmax1 - xmin1, ymax1 - ymin1
        w2, h2 = xmax2 - xmin2, ymax2 - ymin2
        center_x1 = (xmax1 + xmin1) / 2
        center_y1 = (ymax1 + ymin1) / 2
        center_x2 = (xmax2 + xmin2) / 2
        center_y2 = (ymax2 + ymin2) / 2
        inter_diag = (center_x1 - center_x2) / 2 ** 2 + (center_y1 - center_y2) ** 2

    iou = IoU(box1, box2, wh)
    enclose1 = max(max(xmax1, xmax2) - min(xmin1, xmin2), 0.0)
    enclose2 = max(max(ymax1, ymax2) - min(ymin1, ymin2), 0.0)
    outer_diag = (enclose1 ** 2) + (enclose2 ** 2)
    u = (inter_diag) / outer_diag

    arctan = math.atan(w2 / h2) - math.atan(w1 / h1)
    v = (4 / (math.pi ** 2)) * (math.atan(w2 / h2) - math.atan(w1 / h1)) ** 2
    S = 1 - iou
    alpha = v / (S + v)
    w_temp = 2 * w1
    distance = w1 ** 2 + h1 ** 2
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    if not normaled:
        cious = iou - (u + alpha * ar / distance)
    else:
        cious = iou - (u + alpha * ar)
    cious = np.clip(cious, a_min=-1.0, a_max=1.0)

    return cious


def bbox_giou_np(boxes1, boxes2):
    # xywh -> xyxy
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = np.concatenate([np.minimum(boxes1[..., :2], boxes1[..., 2:]),
                             np.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = np.concatenate([np.minimum(boxes2[..., :2], boxes2[..., 2:]),
                             np.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    # 计算最小闭合凸面 C 的面积
    enclose_area = enclose[..., 0] * enclose[..., 1]
    # 根据 GIoU 公式计算 GIoU 值
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


# https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/4d4a403d00e6e887ecb7229719b1407d2e132811/4-Object_Detection/YOLOV3/core/yolov3.py#L121
def bbox_giou_tf(boxes1, boxes2):
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    # 计算最小闭合凸面 C 的面积
    enclose_area = enclose[..., 0] * enclose[..., 1]
    # 根据 GIoU 公式计算 GIoU 值
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def bbox_giou_torch(boxes1, boxes2):
    # boxes1, boxes2 = torch.tensor(boxes1, dtype=torch.float32), torch.tensor(boxes2, dtype=torch.float32)
    boxes1, boxes2 = torch.from_numpy(boxes1).float(), torch.from_numpy(boxes2).float()
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = torch.max(right_down - left_up, torch.tensor(0.0))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.tensor(0.0))
    # 计算最小闭合凸面 C 的面积
    enclose_area = enclose[..., 0] * enclose[..., 1]
    # 根据 GIoU 公式计算 GIoU 值
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


# https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/65b68b53f73173397937d4950ff916a41545c960/utils/box/box_utils.py#L5
def bbox_diou_torch(bboxes1, bboxes2):
    bboxes1, bboxes2 = torch.from_numpy(bboxes1).float(), torch.from_numpy(bboxes2).float()
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]  # 交集
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area  # 并集
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


def bbox_diou_np(boxes1, boxes2, normaled=False):
    inter_diag = np.sum(np.square(boxes1[..., :2] - boxes2[..., :2]), axis=1)
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = np.concatenate([np.minimum(boxes1[..., :2], boxes1[..., 2:]),
                             np.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = np.concatenate([np.minimum(boxes2[..., :2], boxes2[..., 2:]),
                             np.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    outer_diag = (enclose[:, 0] ** 2) + (enclose[:, 1] ** 2)
    # 根据 DIoU 公式计算 DIoU 值
    diou = iou - 1.0 * inter_diag / outer_diag
    diou = np.clip(diou, a_min=-1.0, a_max=1.0)

    return diou


def bbox_diou_tf(boxes1, boxes2):
    inter_diag = tf.reduce_sum(tf.square(boxes1[..., :2] - boxes2[..., :2]), axis=1)
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    outer_diag = (enclose[:, 0] ** 2) + (enclose[:, 1] ** 2)
    # 根据 GIoU 公式计算 GIoU 值
    diou = iou - 1.0 * inter_diag / outer_diag
    diou = tf.clip_by_value(diou, clip_value_min=-1.0, clip_value_max=1.0)

    return diou


# https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/65b68b53f73173397937d4950ff916a41545c960/utils/box/box_utils.py#L47
def bbox_ciou_torch(bboxes1, bboxes2, normaled=False):
    bboxes1, bboxes2 = torch.from_numpy(bboxes1).float(), torch.from_numpy(bboxes2).float()
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
        distance = w1 ** 2 + h1 ** 2
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    if not normaled:
        cious = iou - (u + alpha * ar / distance)
    else:
        cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious


def bbox_ciou_np(boxes1, boxes2, normaled=False):
    w1, h1 = boxes1[..., 2], boxes1[..., 3]
    w2, h2 = boxes2[..., 2], boxes2[..., 3]
    inter_diag = np.sum(np.square(boxes1[..., :2] - boxes2[..., :2]), axis=-1)
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = np.concatenate([np.minimum(boxes1[..., :2], boxes1[..., 2:]),
                             np.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = np.concatenate([np.minimum(boxes2[..., :2], boxes2[..., 2:]),
                             np.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    outer_diag = (enclose[:, 0] ** 2) + (enclose[:, 1] ** 2)
    u = (inter_diag) / outer_diag
    # 根据 CIoU 公式计算 CIoU 值
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (math.pi ** 2)) * np.square(np.arctan(w2 / h2) - np.arctan(w1 / h1))
    S = 1 - iou
    alpha = v / (S + v)
    w_temp = 2 * w1
    distance = w1 ** 2 + h1 ** 2
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    if not normaled:
        cious = iou - (u + alpha * ar / distance)
    else:
        cious = iou - (u + alpha * ar)
    cious = np.clip(cious, a_min=-1.0, a_max=1.0)

    return cious


def bbox_ciou_tf(boxes1, boxes2, normaled=False):
    w1, h1 = boxes1[..., 2], boxes1[..., 3]
    w2, h2 = boxes2[..., 2], boxes2[..., 3]
    inter_diag = tf.reduce_sum(tf.square(boxes1[..., :2] - boxes2[..., :2]), axis=-1)
    # pred_xywh, label_xywh -> pred_xyxy, label_xyxy
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # 计算两个边界框之间的 iou 值
    iou = inter_area / union_area
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    # 计算最小闭合凸面 C 左上角和右下角的坐标
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    outer_diag = (enclose[:, 0] ** 2) + (enclose[:, 1] ** 2)
    u = (inter_diag) / outer_diag
    # 根据 CIoU 公式计算 CIoU 值
    # arctan = tf.atan(w2 / h2) - tf.atan(w1 / h1)
    # v = (4 / (math.pi ** 2)) * np.square(tf.atan(w2 / h2) - tf.atan(w1 / h1))
    arctan = tf.atan(w2 / (h2 + epsilon)) - tf.atan(w1 / (h1 + epsilon))
    v = (4 / (math.pi ** 2)) * np.square(tf.atan(w2 / (h2 + epsilon)) - tf.atan(w1 / (h1 + epsilon)))
    S = 1 - iou
    alpha = tf.stop_gradient(v / (S + v))
    w_temp = tf.stop_gradient(2 * w1)
    distance = tf.stop_gradient(w1 ** 2 + h1 ** 2 + epsilon)
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    if not normaled:
        cious = iou - (u + alpha * ar / distance)
    else:
        cious = iou - (u + alpha * ar)
    cious = tf.clip_by_value(cious, clip_value_min=-1.0, clip_value_max=1.0)

    return cious
