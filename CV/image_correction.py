"""用于对图像内容的矫正"""
import os
import copy
import cv2 as cv
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def transform(image_path, re_height=None, re_width=None):
    image = cv.imread(image_path)
    resize_image, ratio = resize_keep_ratio(image, re_height, re_width)  # resize
    gray_image = cv.cvtColor(resize_image, cv.COLOR_BGR2GRAY)  # 1、灰度化
    blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)  # 2、高斯模糊图
    edged = cv.Canny(blur_image, threshold1=45, threshold2=80)  # 3、边缘检测

    print('Step 1: 边缘检测')
    cv.imshow('edged_image', edged)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 2、筛选轮廓
    screen_contour = det_contours_and_sorted(edged)[0]
    print("Step 2: 轮廓检测")
    drawed_image = cv.drawContours(resize_image, [screen_contour], -1, (0, 255, 0), 2)
    cv.imshow('drawed_image', drawed_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 3、矫正图像
    print("Step 3: 矫正图像")
    warped_image = correction_transform(image, np.array(screen_contour).reshape(-1, 2) / ratio)
    cv.imshow('warped image', warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 3、文字识别
    gray_warped = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
    _, bin_warped = cv.threshold(gray_warped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('bin warped image', bin_warped)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('bin_warp_image.jpg', warped_image)

    pillow_image = Image.open('bin_warp_image.jpg')
    text = pytesseract.image_to_string(pillow_image)
    print('识别出的字体如下: \n', text)
    return warped_image


def det_contours_and_sorted(image, max_counts=4):
    """轮廓检测, 轮廓按照面积大小进行排序
    image: 最好是边缘检测后的图像
    max_counts: 提取图像中多少个矩形
    """
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  # 统计图像轮廓
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:(max_counts + 1)]  # 取前四个最大的轮廓
    rectangle_contours = []  # 用于保存只有轮廓是矩形的轮廓
    for contour in contours:
        peri = cv.arcLength(contour, True)  # 计算轮廓的长度
        # 计算出轮廓的近似，因为有些轮廓并不是由完整的线组成，可能是断断续续的点组成
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # 当approx包含四个点的时候返回，即矩形需要保存
            rectangle_contours.append(approx)
    return rectangle_contours


def order_points(pts):
    """将坐标点"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角的点坐标和应该是最小的
    rect[2] = pts[np.argmax(s)]  # 右下角的坐标和应该是最大的

    # # 沿着指定轴计算第N维的离散差值, 从输出结果可以看出，其实diff函数就是执行的是后一个元素减去前一个元素。
    diff = np.diff(pts, axis=1)  # 计算坐标相似度,
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def correction_transform(image, roi_points):
    """
    :param image: 为原始图像
    :param roi_points: 用于矫正图像中内容四个点的坐标
    :return:
    """
    rect = order_points(roi_points)
    # rect = np.array([[64.8, 969.6],
    #                  [528., 556.8],
    #                  [830.4, 804.],
    #                  [434.4, 1300.8],
    #                  ], dtype=np.float32)
    tl, tr, br, bl = rect  # top_left、top_right、bottom_right、bottom_left

    # 1、首先计算边框的宽度和长度
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # sqrt(x**2 + y**2)
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # 另一条宽
    width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # sqrt(x**2 + y**2)
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # 另一条宽
    height = max(int(height_a), int(height_b))

    # 2、变换后对应坐标的位置: 上左、上右、下右、下左
    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float32)

    # 3、计算变换矩阵
    transform_matrix = cv.getPerspectiveTransform(rect, dst)
    warped_image = cv.warpPerspective(image, transform_matrix, (width, height))
    return warped_image


def resize_keep_ratio(image, re_height=None, re_width=None, inter=cv.INTER_AREA):
    """对图像进行resize操作，但是不改变图像长、宽比例
    :return: ratio--resize后与原始图像长宽之间的比例
    """
    if re_width is None:
        ratio = re_height / image.shape[0]  # 图像resize的比例
        re_width = int(ratio * image.shape[1])  # 根据同样的比例缩放宽
    elif re_height is None:
        ratio = re_width / image.shape[1]  # 图像resize的比例
        re_height = int(ratio * image.shape[0])
    else:
        ratio = 1.
        return image, ratio
    result = copy.deepcopy(image)
    result = cv.resize(result, (re_width, re_height), interpolation=inter)
    return result, ratio


if __name__ == '__main__':
    transform(r'D:\Users\YingYing\Desktop\english.png', 800)
