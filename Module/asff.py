from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

""" ASFF: 对不同层的特征图，在整合的时候，给予不同的权重"""


def conv_bn_activate(inputs, filters, k_size, strides, activate='lrelu'):
    out = layers.Conv2D(filters=filters, kernel_size=k_size, strides=strides,
                        padding='same', use_bias=False)(inputs)
    out = layers.BatchNormalization()(out)
    if activate == 'lrelu':
        out = tf.nn.leaky_relu(out, 0.1)
    else:
        out = tf.nn.relu6(out)
    return out


def asff_layer(inputs, level, rfb=False, vis=False):
    """
    :param inputs: 在原始论文中，x_level_0特征图分辨率最小，通道数为512,
                   x_level_1 和 x_level_2 特征图分辨率逐级增大，通道数为 256
    :param level: 在 ASFF中，有三种尺度的特征图，对应三个不同 level的输出，level-0的分辨率最小
    :param rfb: 是否需要减少内存消耗
    :param vis: 是否输出 权重矩阵
    :return: 在最后输出的时候，level 0-2 特征图的通道数分别为: 1024、512、256
    """
    x_level_0, x_level_1, x_level_2 = inputs
    compress_c = 8 if rfb else 16  # 在使用到 rfb 的时候，将使用的内存减半

    if level == 0:
        inter_dim = 512  # 等于对应 level 特征图的通道数
        expand_dim = 1024  # 最后输出特征图的通道数
        level_0_resized = x_level_0
        level_1_resized = conv_bn_activate(x_level_1, filters=inter_dim, k_size=3, strides=2)
        level_2_downsample = layers.MaxPooling2D(3, 2, padding='same')(x_level_2)
        level_2_resized = conv_bn_activate(level_2_downsample, filters=inter_dim, k_size=3, strides=2)
    elif level == 1:
        inter_dim = 256
        expand_dim = 512
        level_0_compressed = conv_bn_activate(x_level_0, inter_dim, k_size=1, strides=1)
        level_0_resized = layers.UpSampling2D()(level_0_compressed)
        level_1_resized = x_level_1
        level_2_resized = conv_bn_activate(x_level_2, inter_dim, k_size=3, strides=2)
    elif level == 2:
        inter_dim = 256
        expand_dim = 256
        level_0_compressed = conv_bn_activate(x_level_0, inter_dim, k_size=1, strides=1)
        level_0_resized = layers.UpSampling2D((4, 4))(level_0_compressed)
        level_1_resized = layers.UpSampling2D()(x_level_1)
        level_2_resized = x_level_2

    level_0_weight_v = conv_bn_activate(level_0_resized, filters=compress_c, k_size=1, strides=1)
    level_1_weight_v = conv_bn_activate(level_1_resized, filters=compress_c, k_size=1, strides=1)
    level_2_weight_v = conv_bn_activate(level_2_resized, filters=compress_c, k_size=1, strides=1)
    level_concat = layers.Concatenate()([level_0_weight_v, level_1_weight_v, level_2_weight_v])
    level_weights = layers.Softmax()(
        layers.Conv2D(filters=3, kernel_size=1, strides=1)(level_concat)
    )
    fused_out_reduced = level_0_resized * level_weights[..., 0:1] + \
                        level_1_resized * level_weights[..., 1:2] + \
                        level_2_resized * level_weights[..., 2:]
    out = conv_bn_activate(fused_out_reduced, filters=expand_dim, k_size=3, strides=1)
    if vis:
        return out, level_weights, tf.math.reduce_sum(out, axis=-1)
    else:
        return out


if __name__ == '__main__':
    inputs_ = keras.Input((32, 32, 512))
    out_0 = conv_bn_activate(inputs_, filters=256, k_size=3, strides=1)
    out_1 = conv_bn_activate(out_0, filters=256, k_size=3, strides=2)
    out_2 = conv_bn_activate(out_1, filters=512, k_size=3, strides=2)
    out_ = asff_layer(inputs=[out_2, out_1, out_0], level=1)
    model_ = keras.Model(inputs_, out_)
    model_.summary()
    print(out_.shape)
