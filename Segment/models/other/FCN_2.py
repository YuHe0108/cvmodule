import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(inputs, filters, conv_repeats, index):
    """多个卷积层 + 一个 MaxPool2d"""
    out = inputs
    for i in range(conv_repeats):
        out = layers.Conv2D(
            filters, kernel_size=3, padding='same', activation='relu', name=f'conv{index}_{i + 1}')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f'pool{index}')(out)
    return out


def fcn_8_helper(image_size, num_classes):
    inputs = keras.Input(image_size)

    # 一共下采样了 5 次, 也就是 32 倍下采样。
    down_out = conv_block(inputs, 64, 2, index=1)
    down_out = conv_block(down_out, 128, 2, index=2)
    down_out = conv_block(down_out, 256, 3, index=3)
    down_out = conv_block(down_out, 512, 3, index=4)
    down_out = conv_block(down_out, 512, 5, index=5)

    out = layers.Conv2D(1024, kernel_size=7, padding='same', activation='relu', name='fc6')(down_out)
    out = layers.Conv2D(1024, kernel_size=1, padding='same', activation='relu', name='fc7')(out)
    # score_fr 可以直接用于分割结果，需要直接对特征图上采样32倍，也就是FCN-32
    out = layers.Conv2D(num_classes, kernel_size=1, padding='same', activation='relu', name='score_fr')(out)

    # 如果输入的分辨率为512的话，那么conv_size=16
    conv_size = out.shape[2]
    # 上采样一次, 分辨率为: 34
    up_out = layers.Conv2DTranspose(num_classes, kernel_size=4, strides=2, padding='valid', name='score_2')(out)
    deconv_size = up_out.shape[2]
    extra = (deconv_size - 2 * conv_size)  # 34 - 2 * 16 = 2

    # 输出为分辨率为 32, 上采样16倍得到预测图, 也就是FCN-16
    out = layers.Cropping2D(cropping=((0, extra), (0, extra)))(up_out)
    return keras.Model(inputs, out, name='fcn8_helper')


def fcn8_model(image_size, num_classes):
    fcn_8 = fcn_8_helper(image_size, num_classes)

    # 32 if image size is 512*512
    conv_size = fcn_8.layers[-1].output_shape[2]
    skip_con1 = layers.Conv2DTranspose(
        num_classes, kernel_size=(1, 1), padding="same", activation=None, name="score_pool4")
    summed = layers.add(inputs=[skip_con1(fcn_8.layers[14].output), fcn_8.layers[-1].output])

    # Upsampling output of first skip connection
    x = layers.Conv2DTranspose(
        num_classes, kernel_size=(4, 4), strides=(2, 2), padding="valid", activation=None, name="score4")(summed)
    x = layers.Cropping2D(cropping=((0, 2), (0, 2)))(x)

    # Conv to be applied to pool3
    skip_con2 = layers.Conv2DTranspose(
        num_classes, kernel_size=(1, 1), padding="same", activation=None, name="score_pool3")
    summed = layers.add(inputs=[skip_con2(fcn_8.layers[10].output), x])

    # Final Up convolution which restores the original image size
    Up = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8),
                                padding="valid", activation=None, name="upsample")(summed)
    final = layers.Cropping2D(cropping=((0, 8), (0, 8)))(Up)

    return keras.Model(fcn_8.inputs, final)


if __name__ == '__main__':
    model_ = fcn8_model((256, 256, 1), 1)
    model_.summary()
