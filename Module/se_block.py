from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

"""
SENet: Squeeze-and-Excitation block, 一种通道方向注意力机制。
cbam: 可以看作是SE-Net上的改进
"""


def se_layer(inputs, filters, r=16):
    """ global + Dense + Dense
    :param inputs: 输入
    :param filters: 全连接层神经元的数目
    :param r: reduction，减少参数量
    :return:
    """
    output = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # [batch_size, channel]
    output = tf.keras.layers.Dense(filters // r, activation='relu')(output)
    output = tf.keras.layers.Dense(filters, activation='sigmoid')(output)
    output = tf.expand_dims(output, 1)  # [batch_size, 1, channel]
    output = tf.expand_dims(output, 1)  # [batch_size, 1, 1, channel]
    return layers.Multiply()([inputs, output])  # [batch_size, H, W, C]


if __name__ == '__main__':
    inputs_ = keras.Input((32, 32, 3))
    outputs = layers.Conv2D(32, 3)(inputs_)
    outputs = se_layer(outputs, 32)
    model_ = keras.Model(inputs, outputs)
    model_.summary()
