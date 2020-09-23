import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf


class MaxPoolingWithArgmax2D(layers.Layer):
    """在做最大值池化的时候，会记录下最大值在特征图中存在的位置"""

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        """返回最大池化层结果的同时，还会返回一个mask，记录了最大值在特征图中的位置"""
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        padding = padding.upper()
        ksize = [1, pool_size[0], pool_size[1], 1]
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)

        argmax = tf.cast(argmax, tf.float32)
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(layers.Layer):
    """根据输入，以及mask，对输入上采样， 是MaxPoolingWithArgmax2D的逆过程"""

    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size
        self.strides = (2, 2)
        self.pool_size = (2, 2)

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        if output_shape is None:
            output_shape = (input_shape[0],
                            input_shape[1] * self.size[0],
                            input_shape[2] * self.size[1],
                            input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat(
            [[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)

        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]

        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        ret_output_shape = updates.get_shape().as_list()
        ret_output_shape = (
            ret_output_shape[0],
            ret_output_shape[1] * self.size[0],
            ret_output_shape[2] * self.size[1],
            ret_output_shape[3])
        ret = tf.reshape(ret, [-1, ret_output_shape[1], ret_output_shape[2], ret_output_shape[3]])
        return ret


def conv_block(inputs, kernels, kernel_size=(3, 3), strides=1, activation_fun='relu', need_bn=True):
    output = layers.Conv2D(
        filters=kernels, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    if need_bn:
        output = layers.BatchNormalization()(output)
    if activation_fun != 'none':
        output = layers.Activation(activation_fun)(output)
    return output


def encode_block(inputs, kernels, kernel_size=(3, 3), strides=1, activatio_fun='relu', conv_nums=2):
    """多个卷积 + 一个最大池化层, return: 最大池化层后的特征图，记录了最大值在特征图中位置的mask"""
    outputs = inputs
    for _ in range(conv_nums):
        outputs = layers.Conv2D(
            kernels, kernel_size, strides=strides, activation=activatio_fun, padding='same')(outputs)
    outputs, output_mask = MaxPoolingWithArgmax2D()(outputs)
    return outputs, output_mask


def decode_block(inputs, masks, kernels, kernel_size=(3, 3), down_times=False,
                 strides=1, activation_fun='relu', conv_nums=2):
    """特征图的上采样过程"""
    unpool = MaxUnpooling2D()([inputs, masks])
    output = unpool
    for i in range(conv_nums):
        if down_times and i == conv_nums - 1:
            kernels = kernels // 2
        output = conv_block(output, kernels=kernels, kernel_size=kernel_size, strides=strides,
                            activation_fun=activation_fun)
    return output


def segnet_model(input_shape, DIM=64, n_classes=1):
    inputs = layers.Input(input_shape)

    # 下采样过程
    output, mask_1 = encode_block(inputs, DIM * 1, conv_nums=2)
    output, mask_2 = encode_block(output, DIM * 2, conv_nums=2)
    output, mask_3 = encode_block(output, DIM * 4, conv_nums=3)
    output, mask_4 = encode_block(output, DIM * 8, conv_nums=3)
    output, mask_5 = encode_block(output, DIM * 8, conv_nums=3)

    # 上采样过程
    output = decode_block(output, mask_5, kernels=DIM * 8, conv_nums=3)
    output = decode_block(output, mask_4, kernels=DIM * 8, conv_nums=3, down_times=True)
    output = decode_block(output, mask_3, kernels=DIM * 4, conv_nums=3, down_times=True)
    output = decode_block(output, mask_2, kernels=DIM * 2, conv_nums=2, down_times=True)
    unpool = MaxUnpooling2D()([output, mask_1])
    output = conv_block(unpool, kernels=DIM)

    # 最后输出层
    output = conv_block(output, kernels=n_classes, kernel_size=(1, 1), need_bn=False, activation_fun='none')
    if n_classes == 1:
        outputs = layers.Activation('sigmoid')(output)
    else:
        outputs = layers.Softmax()(outputs)
    return keras.Model(inputs, outputs, name='SegNet')


if __name__ == '__main__':
    seg_model_ = segnet_model(input_shape=(256, 256, 1))
    seg_model_.summary()
