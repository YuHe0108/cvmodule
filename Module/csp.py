"""CSPNET: A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN

1、CSP-Net: Cross Stage Partial Network
    (1) 增强 CNN 的学习能力，使得在轻量化的同时保持准确性。
    (2) 降低计算瓶颈
    (3) 降低内存成本

2、主要特点：
    CSP-Net 将浅层特征映射为两个部分，一部分经过 Dense 模块, 另一部分直接与 Partial Dense Block 输出进行 concat。

3、Reference: CSPNet 论文地址：https://arxiv.org/pdf/1911.11929.pdf
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tf_package.ConvModel import conv_utils


def csp_layer(inputs, filters, compression, groups=2, dense_layer=4):
    """卷积部分采用 Dense-Block 实现"""
    split_outs = tf.split(inputs, num_or_size_splits=groups, axis=-1)
    out = conv_bn_activation(split_outs[1], filters, k_size, padding='same',
                             strides=1, activation='leaky relu', bn=True)
    out = layers.Concatenate()([split_outs[0], out])
    return out
