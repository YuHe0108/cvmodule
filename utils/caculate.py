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


def compute_iou(rec1, rec2):
    """
    computing IoU
    rec1: (x0, y0, x1, y1)
    rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


if __name__ == '__main__':
    res = compute_iou([510, 849, 689, 908], [532, 790, 676, 852])
    print(res)
