# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def calc_iou(box1, box2, wh=False):
#     """计算两个框的 IOU """
#     if wh:
#         xmin1, ymin1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
#         xmax1, ymax1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
#         xmin2, ymin2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
#         xmax2, ymax2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
#     else:
#         xmin1, ymin1, xmax1, ymax1 = box1
#         xmin2, ymin2, xmax2, ymax2 = box2
#
#     # 计算交集部分尺寸
#     W = min(xmax1, xmax2) - max(xmin1, xmin2)
#     H = min(ymax1, ymax2) - max(ymin1, ymin2)
#
#     # 计算两个矩形框面积
#     SA = (xmax1 - xmin1) * (ymax1 - ymin1)
#     SB = (xmax2 - xmin2) * (ymax2 - ymin2)
#
#     cross = max(0, W) * max(0, H)  # 计算交集面积
#     iou = float(cross) / (SA + SB - cross)
#     return iou
#
# img = cv2.cvtColor(plt.imread(r'C:\Users\yuhe\Desktop\shenzhen_val_data\梅林四村其他垃圾0830-03544.jpg'), cv2.COLOR_RGB2BGR)
# coord = ["507 832 752 1076",
#          "1324 904 1389 1034",
#          "513 633 942 882",
#          "837 979 978 1080",
#          "835 519 1079 818",
#          "1049 570 1206 745",
#          "1116 402 1341 598",
#          "943 741 1346 1080"]
#
# target = ["507 632 908 896",
#           "1038 574 1206 739",
#           "1119 396 1339 592",
#           "954 745 1333 1080",
#           "836 517 1074 823"]
# for i, info in enumerate(coord):
#     x1, y1, x2, y2 = info.split(' ')
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     for j, t_info in enumerate(target):
#         t_x1, t_y1, t_x2, t_y2 = t_info.split(' ')
#         t_x1, t_y1, t_x2, t_y2 = int(t_x1), int(t_y1), int(t_x2), int(t_y2)
#         print(i, j, calc_iou([x1, y1, x2, y2], [t_x1, t_y1, t_x2, t_y2]))
#     if i == 1 or i == 3:
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
#         cv2.imshow('img', img)
#         cv2.waitKey(0)
# import torch
# from Detection.yolov5.models.experimental import attempt_load
#
# model = attempt_load(r'D:\Vortex\SVN\湖州垃圾分类质量\20220902\0902_xj3.pt', map_location=torch.device('cpu'))
# m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
# print(m.anchor_grid)
import cv2


def join_together(img, shape):
    out = img
    h, w = out.shape[:2]
    ori_h, ori_w = img.shape[:2]
    while w < shape[0] or w < shape[1]:
        out = cv2.copyMakeBorder(out, ori_h, 0, ori_w, 0, cv2.BORDER_WRAP)
        h, w = out.shape[:2]
    return out[0:shape[0], 0:shape[1]]

if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\yuhe\Desktop\1.png")
    res = join_together(img, (1000, 1000))
    cv2.imshow('img', res)
    cv2.waitKey(0)