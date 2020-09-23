from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


def sobel(gray, size):
    # ksize是指核的大小,只能取奇数，影响边缘的粗细
    x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=size)
    y = cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=size)

    # 转回uint8
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def watershed(image_path, reshape_size=None):
    # os.chdir(os.path.dirname(__file__))
    img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    if reshape_size:
        img = cv.resize(img, reshape_size)
    # 图片先转成灰度的
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gradient = sobel(gray, 3)
    # opencv的watershed函数只接受三通道的图像，因此需要将灰度图转换到BGR色彩空间
    gradient = cv.cvtColor(gradient, cv.COLOR_GRAY2BGR)

    # 分水岭算法需要对背景、前景（目标）、未知区域进行标记。
    # 标记的数据类型必须是int32，否则后面会报错
    # 未知区域标记为0（黑色）
    markers = np.zeros_like(gray, dtype=np.int32)

    # 确定的背景标记为1
    markers[np.where(gray > 253)] = 1

    # 分离前景区域
    ret, foreground = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

    # 确定的前景目标标记为2、3、4......(不同目标标记为不同序号，方面后面进行粘连前景的分割)
    markers[np.where(foreground == 0)] = 2
    markers2 = markers.copy()

    # 只接受三通道的图像, 分水岭变换的结果会保存在markers2中
    cv.watershed(gradient, markers2)
    # plt.imshow(foreground, cmap='gray')
    # plt.show()
    return foreground


if __name__ == '__main__':
    watershed('brain.jpg')
