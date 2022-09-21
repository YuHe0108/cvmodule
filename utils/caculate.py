def calc_iou(box1, box2, wh=False):
    """计算两个框的 IOU """
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
