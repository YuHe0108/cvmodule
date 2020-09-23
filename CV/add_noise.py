import numpy as np


def saltpepper_noise(inputs, proportion):
    """为输入添加椒盐噪声"""
    image_copy = inputs.copy()
    # 求得其高宽
    img_Y, img_X = inputs.shape
    # 噪声点的 X 坐标
    X = np.random.randint(img_X, size=(int(proportion * img_X * img_Y),))
    # 噪声点的 Y 坐标
    Y = np.random.randint(img_Y, size=(int(proportion * img_X * img_Y),))
    # 噪声点的坐标赋值
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion * img_X * img_Y),))

    # 噪声容器
    sp_noise_plate = np.ones_like(image_copy) * 127
    # 将噪声给噪声容器
    sp_noise_plate[Y, X] = image_copy[Y, X]
    return image_copy, sp_noise_plate  # 这里也会返回噪声，注意返回值
