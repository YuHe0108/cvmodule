"""将bounding_box显示在原始图像上，将人体关键点标注在原始图像上"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def show_bounding_box(image_path, box_str):
    """box str: [x_min, x_max, box_width, box_height]"""
    image = cv.imread(image_path)
    box_list = [float(x) for x in box_str.split()]
    box_array = np.array(box_list).reshape(-1, 4)
    num_boxes = box_array.shape[0]
    for i in range(num_boxes):
        x1, y1 = int(box_array[i][0]), int(box_array[i][1])
        x2, y2 = int(x1 + box_array[i][2]), int(y1 + box_array[i][3])
        image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv.imshow('image', image)
    cv.waitKey(0)
    return


def show_keypoints(image_path, point_str):
    image = cv.imread(image_path)
    point_list = [int(x) for x in point_str.split(' ')]
    point_array = np.array(point_list).reshape(-1, 3)
    num_of_point = point_array.shape[0]
    for i in range(num_of_point):
        image = cv.circle(image, (point_array[i][0], point_array[i][1]), 3, (0, 0, 255), thickness=-1)

    cv.imshow('image', image)
    cv.waitKey(0)
    return


if __name__ == '__main__':
    point_str_ = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 142 309 1 177 320 2 191 398 2 237 " \
                 "317 2 233 426 2 306 233 2 92 452 2 123 468 2 0 0 0 251 469 2 0 0 0 162 551 2"
    show_keypoints(r'J:\DATA\ObjDet\COCO\val2017\000000425226.jpg', point_str_)
    bound_box_str_ = '73.35 206.02 300.58 372.5'
    show_bounding_box(r'J:\DATA\ObjDet\COCO\val2017\000000425226.jpg', bound_box_str_)

# J:\DATA\ObjDet\COCO\val2017\000000425226.jpg
# 640 480 73.35 206.02
# 300.58 372.5
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 142 309 1 177 320 2 191 398 2 237 317 2 233 426 2 306 233 2 92 452 2 123 468 2 0 0 0 251 469 2 0 0 0 162 551 2
