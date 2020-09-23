import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


def get_norm_layer(norm_name, **kwargs):
    norm_name = str(norm_name).lower()
    if norm_name == 'none':
        return keras.layers.Lambda(lambda x: x)
    elif norm_name == 'batch_norm':
        return tf.keras.layers.BatchNormalization(**kwargs)
    elif norm_name == 'layer_norm':
        return tf.keras.layers.LayerNormalization(**kwargs)
    elif norm_name == 'instance_norm':
        return InstanceNorm(**kwargs)
    else:
        raise 'Not implement'


class InstanceNorm(keras.layers.Layer):
    # InstanceNorm就是对一个batch的feature maps, 每一个batch的每一张特征图分别进行归一化
    # BN 是对所有Batch的同一个channel上的特征图进行归一化
    def __init__(self, scale=True, center=True, gamma_initializer=None,
                 beta_initializer=None, epsilon=1e-6, trainable=True):
        super(InstanceNorm, self).__init__()
        self.scale = scale
        self.center = center
        self.epsilon = epsilon
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.trainable = trainable
        self.moments_axes = None  # 归一化的batch所在的维度
        self.gamma = None
        self.beta = None

        if self.gamma_initializer is None:
            self.gamma_initializer = tf.ones_initializer()
        if self.beta_initializer is None:
            self.beta_initializer = tf.zeros_initializer()

    def build(self, input_shape):
        # input_shape 输入BN数据的shape: [bz, h, w, c]
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank: {}'.format(input_shape))
        n_dims = len(input_shape)
        assert n_dims > 0

        reduction_axis = n_dims - 1  # channel这个维度 reduction_axis=3
        moments_axis = list(range(n_dims))  # moments_axis=[0, 1, 2, 3]
        del moments_axis[reduction_axis]  # moments_axis=[0, 1, 2]
        del moments_axis[0]  # moments_axis=[1, 2]
        self.moments_axes = tf.convert_to_tensor(moments_axis)
        param_shape = input_shape[reduction_axis:reduction_axis + 1]  # c

        if self.scale:
            self.gamma = self.add_weight(name='gamma', shape=param_shape,
                                         initializer=self.gamma_initializer,
                                         trainable=self.trainable)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(name='beta', shape=param_shape,
                                        initializer=self.beta_initializer,
                                        trainable=self.trainable)
        else:
            self.beta = None

    def call(self, inputs):
        # 计算均值和方差
        mean, variance = tf.nn.moments(inputs, self.moments_axes, keepdims=True)
        # 返回归一化后的值
        outputs = tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.scale, self.epsilon)
        return outputs
