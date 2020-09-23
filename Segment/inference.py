from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


# from model import deeplab_v3

def normalize_image(inputs, norm_range='00'):
    if norm_range == '-11':
        norm_image = (inputs / 127.5) - 1
    elif norm_range == '00':
        norm_image = inputs / 255.0
    else:
        pass
    return norm_image


def resize_image(inputs, resize_value):
    w, h = inputs.shape[:2]
    ratio = float(resize_value) / np.max([w, h])  # 对最长的一条边
    resized_image = np.array(Image.fromarray(inputs.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

    pad_x = int(resize_value - resized_image.shape[0])  # w
    pad_y = int(resize_value - resized_image.shape[1])  # h
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
    return resized_image, (pad_x, pad_y)


def visualize_result(img_path, model, model_weights_path, resize_value=512, norm_range='00'):
    image = np.array(Image.open(img_path))
    resized_image, (pad_x, pad_y) = resize_image(image, resize_value)
    norm_image = normalize_image(resized_image, norm_range)

    # 加载模型
    model.load_weights(model_weights_path, by_name=True)
    res = model.predict(np.expand_dims(norm_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # 移除图像填充的部分,保证了原始图像的比例不变
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    plt.imshow(labels)
    plt.axes('off')
    plt.show()
    return
