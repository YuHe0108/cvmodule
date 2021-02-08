from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np


# 使用resnet50 作为backbone
def resnet(inputs):
    pass


def aspp_unit(inputs):  # 空间池化金字塔
    dims = keras.backend.int_shape(inputs)

    # 1、一个全局的平均值池化，之后在上采样为原始inputs尺寸
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    x = layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out_pool = tf.keras.layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
                                            interpolation='bilinear')(x)

    # 2、卷积核为 1x1, 输出尺寸没有改变
    x = layers.Conv2D(256, kernel_size=1, dilation_rate=1, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    out_1 = layers.ReLU()(x)

    # 3、卷积核为 3x3, 扩张率为6 的扩张卷积
    x = layers.Conv2D(256, kernel_size=3, dilation_rate=6, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    out_6 = layers.ReLU()(x)

    # 4、卷积核为 3x3, 扩张率为12
    x = layers.Conv2D(256, kernel_size=3, dilation_rate=12, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    out_12 = layers.ReLU()(x)

    # 5、卷积核为 3x3, 扩张率为18
    x = layers.Conv2D(256, kernel_size=3, dilation_rate=18, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    out_18 = layers.ReLU()(x)

    # concat 所有层的输出
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    x = layers.Conv2D(256, kernel_size=1, dilation_rate=1, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    output = layers.ReLU()(x)
    return output


def deeplab_v3_plus(input_shape, n_classes):
    img_h, img_w = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape)
    resnet = keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    x = resnet.get_layer('conv4_block6_out').output
    # x = resnet50.get_layer('conv4_block6_2_relu').output # (None, 16, 16, 256)
    x = aspp_unit(x)  # 经过resnet50 提取之后的特征，经过aspp
    # 四倍上采样
    input_a = layers.UpSampling2D(size=(img_h // 4 // x.shape[1], img_w // 4 // x.shape[2]),
                                  interpolation='bilinear')(x)

    # resnet 中间层特征图
    input_b = resnet.get_layer('conv2_block3_out').output
    input_b = layers.Conv2D(256, kernel_size=(1, 1), padding='same', use_bias=False,
                            kernel_initializer=tf.keras.initializers.he_normal())(input_b)
    input_b = layers.BatchNormalization()(input_b)
    input_b = layers.ReLU()(input_b)

    # 将中间层与backbone的输出concat
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', use_bias=False,
                      kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', use_bias=False,
                      kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 在经过四倍的上采样
    x = layers.UpSampling2D(size=(img_h // x.shape[1], img_w // x.shape[2]), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes, kernel_size=(1, 1), padding='same')(x)
    if n_classes == 1:
        out = keras.activations.softmax(x)
    else:
        out = keras.activations.sigmoid(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


if __name__ == '__main__':
    model = deeplab_v3_plus((256, 256, 3), 1)
    model.summary()
