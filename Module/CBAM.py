import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
论文中指出了，先使用CA，后使用SA  
定义了:
    channel attention output.shape: [b, 1, 1, filters]
    spatial attention output.shape: [b, h, w, 1]
"""


def regularized_padded_conv(*args, **kwargs):
    """  定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'"""
    return layers.Conv2D(
        *args, **kwargs,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        # kernel_regularizer=keras.regularizers.l2(5e-4)
    )


def channel_attention_dense(inputs, filters, ratio=16):
    avg_out = layers.GlobalAveragePooling2D()(inputs)
    max_out = layers.GlobalMaxPool2D()(inputs)
    out = tf.stack([avg_out, max_out], axis=1)
    out = layers.Dense(filters // ratio,
                       kernel_initializer='he_normal',
                       # kernel_regularizer=keras.regularizers.l2(5e-4),
                       use_bias=True,
                       bias_initializer='zeros',
                       activation='relu'
                       )(out)
    out = layers.Dense(filters,
                       kernel_initializer='he_normal',
                       # kernel_regularizer=keras.regularizers.l2(5e-4),
                       use_bias=True,
                       bias_initializer='zeros'
                       )(out)
    out = tf.reduce_sum(out, axis=1)
    out = layers.Activation('sigmoid')(out)
    out = layers.Reshape((1, 1, out.shape[1]))(out)
    return out


def channel_attention_conv(inputs, filters, ratio=16):
    """将全连接层替换为卷积层： channel attention 输出: [B, 1, 1, filters]"""
    avg_out = layers.GlobalAveragePooling2D()(inputs)
    max_out = layers.GlobalMaxPool2D()(inputs)
    avg_out = layers.Reshape((1, 1, avg_out.shape[1]))(avg_out)
    max_out = layers.Reshape((1, 1, max_out.shape[1]))(max_out)
    out = layers.Concatenate(axis=3)([avg_out, max_out])  # [batch_size, 1, 1, dims+dims]

    pool_out = [avg_out, max_out]
    conv_out = []
    for i in range(2):
        out = layers.Conv2D(filters // ratio,
                            kernel_size=1, strides=1,
                            padding='same',
                            # kernel_regularizer=keras.regularizers.l2(5e-4),
                            use_bias=True, activation=tf.nn.relu
                            )(pool_out[i])
        out = layers.Conv2D(filters,
                            kernel_size=1, strides=1, padding='same',
                            # kernel_regularizer=keras.regularizers.l2(5e-4),
                            use_bias=True
                            )(out)
        conv_out.append(out)
    conv_out = conv_out[0] + conv_out[1]
    out = layers.Reshape((1, 1, filters))(out)
    out = layers.Activation('sigmoid')(out)
    return out


class ChannelAttentionConv(layers.Layer):
    def __init__(self, out_filters, ratio=16):
        super(ChannelAttentionConv, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(
            out_filters // ratio, kernel_size=1, strides=1, padding='same',
            # kernel_regularizer=keras.regularizers.l2(5e-4),
            use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(
            out_filters, kernel_size=1, strides=1, padding='same',
            # kernel_regularizer=keras.regularizers.l2(5e-4),
            use_bias=True)

    def build(self, input_shape):
        filter_size = input_shape[1]
        input_filters = input_shape[-1]
        self.conv_filter_size = layers.Conv2D(
            input_filters, kernel_size=filter_size, strides=1, padding='valid',
            # kernel_regularizer=keras.regularizers.l2(5e-4),
            use_bias=True)
        return

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1, max.shape[1]))(max)  # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        return out


class ChannelAttentionDense(layers.Layer):
    """channel attention 自定义类"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionDense, self).__init__()

        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(in_planes // ratio,
                                kernel_initializer='he_normal',
                                # kernel_regularizer=keras.regularizers.l2(5e-4),
                                use_bias=True,
                                bias_initializer='zeros',
                                activation='relu')
        self.fc2 = layers.Dense(in_planes,
                                kernel_initializer='he_normal',
                                # kernel_regularizer=keras.regularizers.l2(5e-4),
                                use_bias=True,
                                bias_initializer='zeros')

    def build(self, input_shape):
        pass

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg(inputs)))
        max_out = self.fc2(self.fc1(self.max(inputs)))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        out = layers.Reshape((1, 1, out.shape[1]))(out)
        return out


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=1, kernel_size=kernel_size, strides=1, activation='sigmoid',
            padding='same', use_bias=False, kernel_initializer='he_normal',
            # kernel_regularizer=keras.regularizers.l2(5e-4)
        )

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)  # [b, h, w, 1]
        max_out = tf.reduce_max(inputs, axis=3)  # [b, h, w, 1]
        out = tf.stack([avg_out, max_out], axis=-1)  # 创建一个维度,拼接到一起concat。[b, h, w, 2]
        out = self.conv1(out)  # [b, h, w, 1]
        return out


def test_model(input_shape):
    inputs = layers.Input(input_shape)
    out = SpatialAttention()(inputs)
    return tf.keras.Model(inputs, out)


if __name__ == '__main__':
    model_ = test_model((32, 32, 64))
    model_.summary()
