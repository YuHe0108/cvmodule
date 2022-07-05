import numpy as np
import cv2


def image_process(img, value_range, input_format='channel_first'):
    """ 将图像归一化至 01（-1, 1）之间，并转换 HWC ~ CHW, 增加维度： BCHW"""
    inputs = img.astype(np.float32)
    if value_range == '01':  # 将 img 的值归一化至 0-1 之间
        inputs /= 255
    else:  # -1 ~ 1 之间
        inputs = inputs / 127.5 - 1

    if input_format == 'channel_first':
        inputs = np.transpose(inputs, [2, 0, 1])
    return np.expand_dims(inputs, 0)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """缩放图像尺寸至： new_shape"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im
