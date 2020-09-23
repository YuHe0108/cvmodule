from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math
import tf_package

# label_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
#                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#                'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
#                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', ]  # noqa

label_names = ['dog', 'cat']  # 背景没有算在内


def get_color(c, x, max_value, colors=None):
    """获得rgb各个分量的颜色值
    :param c:
    :param x:
    :param max_value:
    :param colors:
    :return:
    """
    if colors is None:
        colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (x / max_value) * 5
    i = math.floor(ratio)  # 向下取整
    j = math.ceil(ratio)  # 向上取整
    ratio -= i
    r = (1. - ratio) * colors[i][c] + ratio * colors[j][c]
    return r


def get_rgb_color(cls, clses):
    """
    :param cls: 图像的类别在label_names下的坐标（index）
    :param clses: 类别名称的长度 len(dog)
    :return:
    """
    offset = cls * 123457 % clses  # 各个分量能取到的最大值
    red = get_color(2, offset, clses)
    green = get_color(1, offset, clses)
    blue = get_color(0, offset, clses)
    return int(red * 255), int(green * 255), int(blue * 255)


class Drawer:
    def __init__(self, font_size=24, font="assets\\Roboto-Regular.ttf", char_width=14):
        self.label_names = label_names

        # 设置字体
        self.font_size = font_size
        self.font = ImageFont.truetype(font, font_size)
        self.char_width = char_width

        # 绘制关键点使用
        self.num_joints = 17
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]
        self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
        self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255)]

    def draw_pose(self, img, kps):
        """Draw the pose like https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/debugger.py#L191
        Arguments
          img: uint8 BGR
          kps: (17, 2) keypoint [[x, y]] coordinates
        """
        kps = np.array(kps, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            cv2.circle(img, (kps[j, 0], kps[j, 1]), 3, self.colors_hp[j], -1)
        for j, e in enumerate(self.edges):
            if kps[e].min() > 0:
                cv2.line(img, (kps[e[0], 0], kps[e[0], 1]), (kps[e[1], 0], kps[e[1], 1]), self.ec[j], 2,
                         lineType=cv2.LINE_AA)
        return img

    def draw_box(self, img, x1, y1, x2, y2, cl):
        cl = int(cl)  # 绘制的边框属于哪一类目标, 整数index

        # 边框左上角: (x1, y1), 右下角: (x2, y2), 四舍五入为整数
        x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
        h = img.shape[0]  # 图像的高
        width = max(1, int(h * 0.006))  # h=1000时，w=6
        name = self.label_names[cl].split()[-1]  # 如果目标名称有两个单词，只选择最后一个单词作为目标名称

        # 根据名称选择颜色
        bgr_color = get_rgb_color(cl, len(self.label_names))[::-1]

        # 绘制边框
        cv.rectangle(img, (x1, y1), (x2, y2), bgr_color, width)

        # 设置字体背景
        font_width = len(name) * self.char_width
        cv.rectangle(img, (x1 - math.ceil(width / 2), y1 - self.font_size), (x1 + font_width, y1), bgr_color, -1)

        # 添加文本
        pil_img = Image.fromarray(img[..., ::-1])
        draw = ImageDraw.Draw(pil_img)
        draw.text((x1 + width, y1 - self.font_size), name, font=self.font, fill=(0, 0, 0, 255))
        img = np.array(pil_img)[..., ::-1].copy()
        return img


if __name__ == '__main__':
    drawer = Drawer()
    img_ = plt.imread(r'J:\DATA\OxfordCat\images\images\Abyssinian_1.jpg')
    return_img = drawer.draw_box(img_, 333, 72, 425, 158, 0)
    plt.imshow(return_img)
    plt.axis('off')
    plt.show()
