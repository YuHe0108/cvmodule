from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

"""定义了一些常见的卷积操作"""


def conv_layer(inputs,
               filters,
               k_size=3,
               strides=1,
               dilate_rate=1,
               use_bias=False):
    """
    当strides=1时， 直接使用padding=‘same’，不该变特征图的大小的卷积层 (当rate>1的时候，为扩张卷积)
    当strides=2时，先填充特征图，之后使用padding=‘valid’，将特征图的尺寸缩放为原来的二分之一。
    """
    if strides == 1:
        return layers.Conv2D(filters,
                             k_size,
                             stride,
                             padding='same',
                             use_bias=use_bias,
                             dilation_rate=dilate_rate,
                             kernel_initializer=keras.initializers.he_normal()
                             )(inputs)
    else:
        k_size_effective = k_size + (k_size - 1) * (dilate_rate - 1)
        pad_total = k_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = layers.ZeroPadding2D((pad_beg, pad_end))(inputs)
        outputs = layers.Conv2D(filters,
                                k_size,
                                strides,
                                padding='valid',
                                use_bias=use_bias,
                                dilation_rate=dilate_rate,
                                kernel_initializer=keras.initializers.he_normal()
                                )(inputs)
        return outputs


def conv_bn_activation(inputs,
                       filters,
                       k_size,
                       padding='same',
                       strides=1,
                       activation='none',
                       bn=True,
                       k_initializer=keras.initializers.he_normal()):
    """按照 conv + bn + relu 的方式组合"""
    activation = str.lower(activation)
    output = layers.Conv2D(filters=filters,
                           kernel_size=k_size,
                           strides=strides,
                           padding=padding,
                           use_bias=not bn,
                           kernel_initializer=k_initializer)(inputs)
    if bn:
        output = layers.BatchNormalization()(output)
    if activation != 'none':
        output = layers.Activation(activation)(output)
    return output


def bn_activation_conv(inputs,
                       filters,
                       k_size,
                       bn=True,
                       strides=1,
                       padding='same',
                       activation='none',
                       k_initializer=keras.initializers.he_normal()):
    """按照 bn + relu + conv 的方式组合"""
    activation = str.lower(activation)
    output = inputs
    if bn:
        output = layers.BatchNormalization()(output)
    if activation != 'none':
        output = layers.Activation(activation)(output)
    output = layers.Conv2D(filters=filters,
                           kernel_size=k_size,
                           strides=strides,
                           padding=padding,
                           kernel_initializer=k_initializer)(output)
    return output


def conv_bn_relu_residual_block(inputs,
                                filters,
                                k_size=3,
                                strides=1,
                                padding='same',
                                activation='relu',
                                k_initializer=keras.initializers.he_normal()
                                ):
    """按照 1x1 3x3 1x1的方式构建残差单元"""

    # 残差单元中间三层
    outputs = conv_bn_activation(inputs,
                                 k_size=1,
                                 filters=filters // 4,
                                 strides=1,
                                 padding=padding,
                                 activation=activation,
                                 bn=True,
                                 k_initializer=k_initializer)
    outputs = conv_bn_activation(outputs,
                                 k_size=k_size,
                                 filters=filters // 4,
                                 strides=strides,
                                 padding=padding,
                                 activation=activation,
                                 bn=True,
                                 k_initializer=k_initializer)
    outputs = conv_bn_activation(outputs,
                                 k_size=1,
                                 filters=filters,
                                 strides=1,
                                 padding=padding,
                                 activation='none',
                                 bn=True,
                                 k_initializer=k_initializer)

    input_dims = inputs.shape[-1]
    # 判断输入和输出的filters 数量是否一样
    if input_dims != filters:
        inputs = conv_bn_activation(inputs, k_size=1, filters=filters, strides=strides, padding=padding,
                                    activation='relu', bn=True, k_initializer=k_initializer)
    outputs = layers.Add()([inputs, outputs])
    outputs = layers.Activation(activation)(outputs)
    return outputs


def bn_relu_conv_residual_block(inputs,
                                filters,
                                k_size=3,
                                strides=1,
                                padding='same',
                                activation='relu',
                                k_initializer=keras.initializers.he_normal()
                                ):
    """按照 1x1 3x3 1x1的方式构建残差单元, 但是顺序依次: BN + ReLU + Conv"""
    # 残差单元中间三层
    outputs = bn_activation_conv(inputs,
                                 k_size=1,
                                 filters=filters // 4,
                                 strides=1,
                                 padding=padding,
                                 activation=activation,
                                 bn=True,
                                 k_initializer=k_initializer)
    outputs = bn_activation_conv(outputs,
                                 k_size=k_size,
                                 filters=filters // 4,
                                 strides=strides,
                                 padding=padding,
                                 activation=activation,
                                 bn=True,
                                 k_initializer=k_initializer)
    outputs = bn_activation_conv(outputs,
                                 k_size=1,
                                 filters=filters,
                                 strides=1,
                                 padding=padding,
                                 activation=activation,
                                 bn=True,
                                 k_initializer=k_initializer)

    input_dims = inputs.shape[-1]
    # 判断输入和输出的filters 数量是否一样
    if input_dims != filters:
        inputs = conv_bn_activation(inputs,
                                    k_size=1,
                                    filters=filters,
                                    strides=strides,
                                    padding=padding,
                                    activation='relu',
                                    bn=True,
                                    k_initializer=k_initializer)
    outputs = layers.Add()([inputs, outputs])
    return outputs


def dense_bottle_neck(inputs,
                      k_size,
                      filters,
                      drop_rate=0.2,
                      strides=1,
                      padding='same',
                      activation='relu',
                      k_initializer=keras.initializers.he_normal()):
    """用于构建DenseBlock"""
    outputs = bn_activation_conv(inputs,
                                 filters=filters * 4,
                                 k_size=1,
                                 strides=1,
                                 padding=padding,
                                 activation=activation,
                                 k_initializer=k_initializer,
                                 )
    outputs = bn_activation_conv(outputs,
                                 filters=filters,
                                 k_size=k_size,
                                 strides=strides,
                                 padding=padding,
                                 activation=activation,
                                 k_initializer=k_initializer,
                                 )
    outputs = layers.Dropout(drop_rate)(outputs)
    return outputs


def dense_block(inputs,
                filters,
                k_size=3,
                strides=1,
                drop_rate=0.2,
                conv_repeats=5,
                padding='same',
                activation='relu',
                k_initializer=keras.initializers.he_normal()
                ):
    """密集形连接
    conv_repeats: dense block 中存在多少个卷积层
    """
    conv_output_list = [inputs]
    output = inputs
    for _ in range(conv_repeats):
        layer_out = dense_bottle_neck(output,
                                      filters=filters,
                                      k_size=k_size,
                                      strides=strides,
                                      padding=padding,
                                      drop_rate=drop_rate,
                                      activation=activation,
                                      k_initializer=k_initializer,
                                      )
        conv_output_list.append(layer_out)
        # 拼接每一个卷积层输出的特征图
        output = tf.concat(conv_output_list, axis=-1)

    conv_output_list.clear()
    return output


def one_concat_block(inputs, filters, k_size=3, conv_repeats=5):
    """将所有的卷积层只拼接一次
    conv_repeats: 中间卷积层的数量
    """
    output_list = [inputs]
    output = inputs
    for _ in range(conv_repeats):
        output = conv_bn_activation(output, filters, k_size=k_size)
        output_list.append(output)
    outputs = tf.concat(output_list, axis=-1)
    return outputs


def inception_block(inputs, filters_list):
    """GoogleNet中提出的 Inception block
    filters_list: 包含了四条支路上的所有卷积层中卷积核的数量
    分为四路:
    第一路: 1x1 卷积
    第二路: 1x1 + 3x3 + 3x3
    第三路: 1x1 + 3x3
    第四路: 1x1 + pooling
    """
    assert type(filters_list) is list
    # 第一条支路
    out_1 = conv_bn_activation(inputs, filters=filters_list[0], k_size=1)

    # 第二条支路
    out_2_1 = conv_bn_activation(inputs, filters=filters_list[1], k_size=1)
    out_2_2 = conv_bn_activation(out_2_1, filters=filters_list[2], k_size=3)

    # 第三条支路
    out_3_1 = conv_bn_activation(inputs, filters=filters_list[3], k_size=1)
    out_3_2 = conv_bn_activation(out_3_1, filters=filters_list[4], k_size=3)
    out_3_3 = conv_bn_activation(out_3_2, filters=filters_list[4], k_size=3)

    # 第四条支路
    out_4_1 = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(inputs)
    out_4_2 = conv_bn_activation(out_4_1, filters=filters_list[5], k_size=3)
    outputs = layers.Concatenate()([out_1, out_2_2, out_3_3, out_4_2])
    return outputs


def depthwise_conv_block(inputs,
                         filters,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         dilate_rate=1,
                         kernel_initializer=keras.initializers.he_normal()
                         ):
    """一个 depthwise + 1x1 卷积"""
    outputs = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=True,
                                     dilation_rate=dilate_rate,
                                     kernel_initializer=kernel_initializer,
                                     )(inputs)
    outputs = conv_bn_activation(outputs,
                                 filters=filters,
                                 k_size=1,
                                 strides=1,
                                 kernel_initializer=kernel_initializer,
                                 padding=padding)
    return outputs


def basic_block_layer(inputs, filters, strides):
    """HANet 中使用到的基础卷积单元: 残差结构: 含有两层卷积层： 1x1 + 3x3"""
    # 判断步长
    if strides != 1:
        residual_output = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same", use_bias=False)(inputs)
        residual_output = layers.BatchNormalization()(residual_output)
    else:
        residual_output = inputs

    # 第一层卷积
    output = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(inputs)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)

    # 第二层卷积
    output = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(output)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)

    output = layers.Add()([residual_output, output])
    output = layers.ReLU()(output)
    return output


def bottle_neck_layer(inputs, filters, strides=1):
    """
    残差单元: 主支路: 1x1 filters + 3x3 filters + 1x1 filrers * 4
    """
    output = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(inputs)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)

    output = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(output)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)

    output = layers.Conv2D(filters * 4, kernel_size=1, strides=1, padding='same', use_bias=False)(output)
    output = layers.BatchNormalization()(output)

    # 侧边支路
    residual_output = layers.Conv2D(filters * 4, kernel_size=1, strides=strides, padding='same', use_bias=False)(inputs)
    residual_output = layers.BatchNormalization()(residual_output)

    output = layers.Add()([residual_output, output])
    output = layers.ReLU()(output)
    return output


def make_basic_layer(inputs, filters, block_nums, strides=1):
    """多个 <两层残差单元> 的叠加
    其中第一个残差单元用于下采样特征图，剩余的用于提取特征
    """
    output = basic_block_layer(inputs, filters, strides=strides)

    for _ in range(1, block_nums):
        output = basic_block_layer(output, filters, strides=1)
    return output


def make_bottleneck_layer(inputs, filters, block_nums, strides=1):
    """多个 <三层残差单元> 的叠加
    其中第一个残差单元用于下采样特征图，剩余的用于提取特征
    """
    output = bottle_neck_layer(inputs, filters, strides=strides)

    for _ in range(1, block_nums):
        output = bottle_neck_layer(output, filters, strides=1)
    return output


if __name__ == '__main__':
    inputs_ = keras.layers.Input((32, 32, 16))
    out = basic_block_layer(inputs_, 16, 2)
    model_ = keras.Model(inputs_, out)
    model_.summary()
