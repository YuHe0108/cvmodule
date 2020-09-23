from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2 as cv
import pathlib
import time
import os

kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)


def residual_conv_unit(inputs, n_filters=256, kernel_size=3, name=''):
    """一个残差单元"""
    net = layers.ReLU(name=name + 'relu1')(inputs)
    net = layers.Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.ReLU(name=name + 'relu2')(net)
    net = layers.Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.Add(name=name + 'sum')([net, inputs])

    return net


def multi_resolution_fusion(high_inputs=None, low_inputs=None, n_filters=256, name=''):
    """多分辨率融合单元, 没有激活函数，如果没有低分辨率特征图输入，则直接输出高分辨率特征图
    """
    if low_inputs is None:  # RefineNet block 4
        return high_inputs
    else:
        # 分辨率低的特征图经过:Conv+BN+UpSampling
        conv_low = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv_lo', kernel_initializer=kern_init,
                                 kernel_regularizer=kern_reg)(low_inputs)
        conv_low = layers.BatchNormalization()(conv_low)
        conv_low_up = layers.UpSampling2D(size=2, interpolation='bilinear', name=name + 'up')(conv_low)

        # 高分辨率特征图经过: Conv+BN
        conv_high = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv_hi', kernel_initializer=kern_init,
                                  kernel_regularizer=kern_reg)(high_inputs)
        conv_high = layers.BatchNormalization()(conv_high)
        return layers.Add(name=name + 'sum')([conv_low_up, conv_high])


def chained_residual_pooling(inputs, n_filters=256, name=''):
    """ 链式残差池化结构: 可以从高分辨率的图像中捕捉上下文背景信息。
    由最大池化层和卷积层组成。
    """
    net = layers.ReLU(name=name + 'relu')(inputs)
    net_out_1 = net

    # 第一层: Conv+BN+MaxPooling
    net = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool1')(net)
    net_out_2 = net

    # 第二层: Conv+BN+MaxPooling
    net = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool2')(net)
    net_out_3 = net

    # 第三层: Conv+BN+MaxPooling
    net = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv3', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool3')(net)
    net_out_4 = net

    # 第四层: Conv+BN+MaxPooling
    net = layers.Conv2D(n_filters, 3, padding='same', name=name + 'conv4', kernel_initializer=kern_init,
                        kernel_regularizer=kern_reg)(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool4')(net)
    net_out_5 = net

    net = layers.Add(name=name + 'sum')([net_out_1, net_out_2, net_out_3, net_out_4, net_out_5])

    return net


def refine_block(high_inputs=None, low_inputs=None, block=0, filters=None):
    """ 一个 refine-block的组成: 残差单元 + 链式池化 + 多分辨率融合
    """

    if low_inputs is None:  # block 4, 第四个block，没有更低分辨率的特征图输入。
        rcu_high = residual_conv_unit(high_inputs, n_filters=filters, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = residual_conv_unit(rcu_high, n_filters=filters, name='rb_{}_rcu_h2_'.format(block))

        # nothing happens here
        fuse = multi_resolution_fusion(high_inputs=rcu_high,
                                       low_inputs=None,
                                       n_filters=filters,
                                       name='rb_{}_mrf_'.format(block))
        fuse_pooling = chained_residual_pooling(fuse, n_filters=filters, name='rb_{}_crp_'.format(block))
        output = residual_conv_unit(fuse, n_filters=filters, name='rb_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = high_inputs.shape[-1]  # 高分辨率特征图filters数量
        low_n = low_inputs.shape[-1]  # 低分辨率特征图filters数量

        # 高分辨率和低分辨率的特征图都先经过两个残差单元
        rcu_high = residual_conv_unit(high_inputs, n_filters=high_n, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = residual_conv_unit(rcu_high, n_filters=high_n, name='rb_{}_rcu_h2_'.format(block))
        rcu_low = residual_conv_unit(low_inputs, n_filters=low_n, name='rb_{}_rcu_l1_'.format(block))
        rcu_low = residual_conv_unit(rcu_low, n_filters=low_n, name='rb_{}_rcu_l2_'.format(block))

        # 之后使用多分辨率特征融合
        fuse = multi_resolution_fusion(high_inputs=rcu_high, low_inputs=rcu_low,
                                       n_filters=filters, name='rb_{}_mrf_'.format(block))

        # 最后使用 <链式池化 + 残差单元> 作为输出
        fuse_pooling = chained_residual_pooling(fuse, n_filters=filters, name='rb_{}_crp_'.format(block))
        output = residual_conv_unit(fuse_pooling, n_filters=filters, name='rb_{}_rcu_o1_'.format(block))
        return output


def build_refine_net(input_shape, num_classes, frontend_trainable=False):
    # Build ResNet-50
    inputs = layers.Input(input_shape)
    model_base = resnet50.ResNet50(input_tensor=inputs, include_top=False,
                                   weights='imagenet', pooling=None, classes=1000)
    output_names = ['conv5_block3_out', 'conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out']
    high = []
    for output_name in output_names:
        high.append(model_base.get_layer(output_name).output)

    # Get ResNet block output layers
    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    for i in range(len(high)):
        if i == 0:
            filters = 256
        else:
            filters = 256
        high[i] = layers.Conv2D(filters, 1, padding='same', name=f'resnet_map{i + 1}',
                                kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[i])
        high[i] = layers.BatchNormalization()(high[i])

    # RefineNet: Only input ResNet 1/32
    low[0] = refine_block(high_inputs=high[0], low_inputs=None, block=4, filters=256)
    # High input = ResNet 1/16, Low input = Previous 1/16
    low[1] = refine_block(high_inputs=high[1], low_inputs=low[0], block=3, filters=256)
    # High input = ResNet 1/8, Low input = Previous 1/8
    low[2] = refine_block(high_inputs=high[2], low_inputs=low[1], block=2, filters=256)
    # High input = ResNet 1/4, Low input = Previous 1/4.
    net = refine_block(high_inputs=high[3], low_inputs=low[2], block=1, filters=256)

    # 输出单元
    net = residual_conv_unit(net, name='rf_rcu_o1_')
    net = residual_conv_unit(net, name='rf_rcu_o2_')
    net = layers.UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(net)
    net = layers.Conv2D(num_classes, 1, name='rf_pred')(net)
    if num_classes == 1:
        net = layers.Activation('sigmoid')(net)
    else:
        net = layers.Activation('softmax')(net)
    model = keras.Model(model_base.input, net, name='RefineNet')

    # 设置模型的参数是否参与训练
    model_base.trainable = frontend_trainable
    # for layer in model.layers:
    #     if 'rb' in layer.name or 'rf_' in layer.name:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = frontend_trainable
    return model


if __name__ == '__main__':
    model_ = build_refine_net(input_shape=(256, 256, 3), num_classes=1, frontend_trainable=False)
    model_.summary()
    print(model_.losses)

# Total params: 58,819,201
# Trainable params: 35,220,225
# Non-trainable params: 23,598,976
