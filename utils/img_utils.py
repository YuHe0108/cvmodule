import numpy as np
import cv2

labels = ['full-trash-bag',
          'plastic-bag',
          'napkin',
          'color-packing',
          'kraft',
          'bottle',
          'can',
          'other']
int_2_label = {i: label for i, label in enumerate(labels)}


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


def draw_img(img, box, color, suffix=''):
    for x1, y1, x2, y2, cls in box:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if color == (0, 0, 255):
            cv2.putText(img, int_2_label[int(cls)] + "-" + suffix, (x1, y1),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        else:
            cv2.putText(img, int_2_label[int(cls)] + "-" + suffix, (x2, y2),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return img


if __name__ == '__main__':
    img = cv2.imread(r'D:\Download\5.jpg')
    img_h, img_w = img.shape[:2]
    box = [[0.7528735632183908, 0.3911290322580645, 0.09051724137931035, 0.3870967741935484]]
    for x, y, w, h in box:
        x = int(x * img_w)
        y = int(y * img_h)
        x2 = x + int(w * img_w)
        y2 = y + int(h * img_h)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
