import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

"""
 PSPNet有很多变种，但是核心的地方在于使用多个池化单元，并将多个池化单元的输出在通道方向上拼接，
 形成特征金子塔。之后经过一个卷积层整合特征，最后使用上采样层直接放大到输入图像的尺寸作为预测图像输出。
"""


def vanilla_encoder(input_shape, dims=64, pad=1, kernel=3, pool_size=2):
    """模型一共下采样了五次，并且返回了每一次下采样的输出特征图"""
    img_input = layers.Input(input_shape)
    x = img_input
    levels = []

    # 1次下采样
    x = layers.ZeroPadding2D((pad, pad))(x)
    x = layers.Conv2D(dims * 1, kernel, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((pool_size, pool_size))(x)
    levels.append(x)

    # 2次下采样
    x = layers.ZeroPadding2D((pad, pad))(x)
    x = layers.Conv2D(dims * 2, kernel, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((pool_size, pool_size))(x)
    levels.append(x)

    # 3、4、5次下采样
    dims = dims * 2
    for _ in range(3):
        dims = dims * 2
        x = layers.ZeroPadding2D((pad, pad))(x)
        x = layers.Conv2D(min(dims, 1024), kernel, padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((pool_size, pool_size), )(x)
        levels.append(x)

    return img_input, levels


def pool_block(inputs, pool_factor):
    h = inputs.shape[1]
    w = inputs.shape[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))
    ]
    x = layers.AveragePooling2D(pool_size, strides=strides, padding='same')(inputs)
    x = layers.Conv2D(512, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D((strides[0], strides[1]), interpolation='bilinear')(x)
    return x


def pspnet(input_shape, n_classes, dims=64, encoder=vanilla_encoder):
    inputs, levels = encoder(input_shape, dims)
    # levels: 2、4、8、16、32被下采样的特征图
    [f1, f2, f3, f4, f5] = levels

    out = f5
    pool_factors = [1, 2, 4, 8]  # 四个不同尺寸的池化层
    pool_outs = [out]
    for p in pool_factors:
        pooled = pool_block(out, p)
        pool_outs.append(pooled)

    # concat: 4个池化层、encoder32倍下采样后输出的特征图
    out = layers.Concatenate()(pool_outs)
    out = layers.Conv2D(1024, (1, 1), use_bias=False, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    out = layers.Conv2D(512, 3, use_bias=False, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    # 输出层 放大8倍
    out = layers.Conv2D(n_classes, 3, padding='same')(out)
    out = layers.UpSampling2D(32, interpolation='bilinear')(out)

    if n_classes == 1:
        out = layers.Activation('sigmoid')(out)
    else:
        out = layers.Softmax()(out)

    model = keras.Model(inputs, out, name='PSPNet')
    return model


if __name__ == '__main__':
    pspnet_ = pspnet(input_shape=(256, 256, 1), n_classes=1, dims=128)
    pspnet_.summary()
