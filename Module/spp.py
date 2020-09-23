from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


def muti_pool_layer(inputs, pool_size):
    """多个不同尺度的池化层,最后将不同最大池化层的输出 concat,
    由于 padding = 'same', 特征图的分辨率没有改变"""
    pool_outs = [inputs]
    for i in range(len(pool_size)):
        pool_out = layers.MaxPool2D(pool_size[i], strides=1, padding='same')(inputs)
        pool_outs.append(pool_out)
    return layers.Concatenate()(pool_outs)


def spp_layer(inputs, pool_size_list, pool_type='max_pool'):
    """SPP(Spatial Pyramid Pooling) 空间金字塔池化
    SPP 显著特点
    1) 不管输入尺寸是怎样，SPP 可以产生固定大小的输出
    2) 使用多个窗口(pooling window)
    3) SPP 可以使用同一图像不同尺寸(scale)作为输入, 得到同样长度的池化特征。

    其它特点
    1) 由于对输入图像的不同纵横比和不同尺寸，SPP同样可以处理，所以提高了图像的尺度不变(scale-invariance)和降低了过拟合(over-fitting)
    2) 实验表明训练图像尺寸的多样性比单一尺寸的训练图像更容易使得网络收敛(convergence)
    3) SPP 对于特定的CNN网络设计和结构是独立的。(也就是说，只要把SPP放在最后一层卷积层后面，对网络的结构是没有影响的， 它只是替换了原来的pooling层)
    4) 不仅可以用于图像分类而且可以用来目标检测
    """
    shape = inputs.shape
    for index, pool_size in enumerate(pool_size_list):
        ksize = [
            1,
            tf.cast(tf.math.ceil(shape[1] / pool_size + 1), tf.int32),
            tf.cast(tf.math.ceil(shape[2] / pool_size + 1), tf.int32),
            1
        ]
        strides = [
            1,
            tf.cast(tf.math.floor(shape[1] / pool_size + 1), tf.int32),
            tf.cast(tf.math.floor(shape[2] / pool_size + 1), tf.int32),
            1
        ]

        if pool_type == 'max_pool':
            pool = tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding='SAME')
            pool = tf.reshape(pool, (-1, pool_size * pool_size * shape[-1]))
        else:
            pool = tf.nn.avg_pool(inputs, ksize=ksize, strides=strides, padding='SAME')
            pool = tf.reshape(pool, (-1, pool_size * pool_size * shape[-1]))

        if index == 0:
            x_flatten = tf.reshape(pool, (-1, pool_size * pool_size * shape[-1]))
        else:
            x_flatten = tf.concat((x_flatten, pool), axis=1)
    return x_flatten


if __name__ == '__main__':
    inputs_ = keras.Input((32, 32, 3))
    out_ = spp_layer(inputs_, (1, 3, 5))
    model_ = keras.Model(inputs_, out_)
    model_.summary()
    model_.save_weights('test')
