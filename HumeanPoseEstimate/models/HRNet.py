from tensorflow import keras
from tensorflow.keras import layers


def conv_bn_layer(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=strides,
                      use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    return x


def basic_block(inputs, filters, strides=1, with_conv_shortcut=False):
    """两层的残差单元"""
    x = conv_bn_layer(inputs, filters, kernel_size=3, strides=strides)
    x = layers.Activation('relu')(x)

    x = conv_bn_layer(x, filters, kernel_size=3)

    if with_conv_shortcut:
        residual = conv_bn_layer(inputs, filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')
        x = layers.Add()([x, residual])
    else:
        x = layers.Add()([x, inputs])

    x = layers.ReLU()(x)
    return x


def bottleneck_block(inputs, filters, strides=1, with_conv_shortcut=False):
    """三层的卷积单元构成的残差单元"""
    expansion = 4
    de_filters = int(filters / expansion)

    # 三层卷积单元
    x = conv_bn_layer(inputs, de_filters, kernel_size=1, strides=1)
    x = layers.ReLU()(x)

    x = conv_bn_layer(x, de_filters, kernel_size=3, strides=strides)
    x = layers.ReLU()(x)

    x = conv_bn_layer(x, filters, kernel_size=1, strides=1)

    if with_conv_shortcut:
        residual = conv_bn_layer(inputs, filters, kernel_size=1, strides=strides)
        x = layers.Add()([x, residual])
    else:
        x = layers.Add()([x, inputs])

    x = layers.ReLU()(x)
    return x


def stem_net(inputs):
    """初始层：一个下采样卷积单元 + 四层残差单元"""
    x = conv_bn_layer(inputs, filters=64, kernel_size=3, strides=2)
    x = layers.ReLU()(x)
    x = conv_bn_layer(x, filters=64, kernel_size=3, strides=2)
    x = layers.ReLU()(x)

    # 因为第一个残差单元的输入和输出的filters数目不一致，因此要侧边支路, 这是包含三个卷积层的残差单元
    x = bottleneck_block(x, 256, with_conv_shortcut=True)
    x = bottleneck_block(x, 256, with_conv_shortcut=False)
    x = bottleneck_block(x, 256, with_conv_shortcut=False)
    x = bottleneck_block(x, 256, with_conv_shortcut=False)
    return x


def make_branch(x, filters):
    """四个由两个卷积层组成的残差单元"""
    x = basic_block(x, filters, with_conv_shortcut=False)
    x = basic_block(x, filters, with_conv_shortcut=False)
    x = basic_block(x, filters, with_conv_shortcut=False)
    x = basic_block(x, filters, with_conv_shortcut=False)
    return x


def transition_layer_1(inputs, filters_list=None):
    """会输出两个尺度的特征图: 原始尺寸的特征图 ,下采样后的特征图"""
    if filters_list is None:
        filters_list = [32, 64]
    x0 = conv_bn_layer(inputs, filters_list[0], kernel_size=3)
    x0 = layers.ReLU()(x0)

    x1 = conv_bn_layer(inputs, filters_list[1], kernel_size=3, strides=2)
    x1 = layers.ReLU()(x1)

    return [x0, x1]


def fuse_layer_1(x):
    """两个平行的支路交叉融合"""
    # 低分辨率特征图上采样 + 高分辨率特征图
    x0_1 = conv_bn_layer(x[1], filters=32, kernel_size=1)
    x0_1 = layers.UpSampling2D()(x0_1)
    x0 = layers.Add()([x[0], x0_1])

    # 高分辨率特征图下采样 + 低分辨率特征图
    x1_0 = conv_bn_layer(x[0], filters=64, kernel_size=3, strides=2)
    x1 = layers.Add()([x1_0, x[1]])
    return [x0, x1]


def transition_layer_2(x, filters_list=None):
    if filters_list is None:
        filters_list = [32, 64, 128]

    """输入含有三个尺度的特征图，并将最后一个下采样"""
    x0 = conv_bn_layer(x[0], filters_list[0], kernel_size=3)
    x0 = layers.ReLU()(x0)

    x1 = conv_bn_layer(x[1], filters_list[1], kernel_size=3)
    x1 = layers.ReLU()(x1)

    x2 = conv_bn_layer(x[1], filters_list[2], kernel_size=3, strides=2)
    x2 = layers.ReLU()(x2)
    return [x0, x1, x2]


def fuse_layer_2(x):
    """特征融合单元: x包含三个尺度特征图的输入
    x[0]: 尺度最大， x[1]: x[0]分辨率的一半， x[2]: x[0]分辨率的四分之一
    """
    # 将两个小尺寸的特征图放大后 与 x[0] 最大的特征图相加
    x0_1 = conv_bn_layer(x[1], filters=32, kernel_size=1)
    x0_1 = layers.UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = conv_bn_layer(x[2], filters=32, kernel_size=1)
    x0_2 = layers.UpSampling2D(size=(4, 4))(x0_2)
    x0 = layers.Add()([x[0], x0_1, x0_2])

    # 将x[0]下采样， x[2]上采样， 与x[1]相加之后作为输出
    x1_0 = conv_bn_layer(x[0], filters=64, kernel_size=3, strides=2)
    x1_2 = conv_bn_layer(x[2], filters=64, kernel_size=1, strides=1)
    x1_2 = layers.UpSampling2D(size=(2, 2))(x1_2)
    x1 = layers.Add()([x1_0, x[1], x1_2])

    # 将x[0]下采样两次， x[1]下采样一次， 与x[2]相加之后作为输出
    x2_0 = conv_bn_layer(x[0], filters=32, kernel_size=3, strides=2)
    x2_0 = layers.ReLU()(x2_0)
    x2_0 = conv_bn_layer(x2_0, filters=128, kernel_size=3, strides=2)
    x2_1 = conv_bn_layer(x[1], filters=128, kernel_size=3, strides=2)
    x2 = layers.Add()([x2_0, x2_1, x[2]])
    return [x0, x1, x2]


def transition_layer_3(x, filters_list=None):
    """输出四个特征图: 前三个特征图的尺寸一致，最后一个特征图的尺寸进行了下采样"""
    if filters_list is None:
        filters_list = [32, 64, 128, 256]
    x0 = conv_bn_layer(x[0], filters_list[0], kernel_size=3)
    x0 = layers.ReLU()(x0)

    x1 = conv_bn_layer(x[1], filters_list[1], kernel_size=3)
    x1 = layers.ReLU()(x1)

    x2 = conv_bn_layer(x[2], filters_list[2], kernel_size=3)
    x2 = layers.ReLU()(x2)

    x3 = conv_bn_layer(x[2], filters_list[3], kernel_size=3, strides=2)
    x3 = layers.ReLU()(x3)
    return [x0, x1, x2, x3]


def fuse_layer_3(x):
    """x包含了四个尺度的特征图，将其余三个小尺度的特征图上采样，与最大尺度的特征图拼接后输出"""
    x0_1 = conv_bn_layer(x[1], filters=32, kernel_size=1)
    x0_1 = layers.UpSampling2D(size=(2, 2))(x0_1)

    x0_2 = conv_bn_layer(x[2], filters=32, kernel_size=1)
    x0_2 = layers.UpSampling2D(size=(4, 4))(x0_2)

    x0_3 = conv_bn_layer(x[3], filters=32, kernel_size=1)
    x0_3 = layers.UpSampling2D(size=(8, 8))(x0_3)

    output = layers.Concatenate()([x[0], x0_1, x0_2, x0_3])
    return output



def seg_hrnet(input_shape, joint_nums):
    inputs = layers.Input(input_shape)

    # 1、初始层
    x = stem_net(inputs)

    # 2-1、中间层: 返回两个尺度的特征图
    x = transition_layer_1(x)
    x0 = make_branch(x[0], 32)
    x1 = make_branch(x[1], 64)
    x = fuse_layer_1([x0, x1])

    # 2-2、中间层: 返回三个尺度的特征图
    x = transition_layer_2(x)
    x0 = make_branch(x[0], 32)
    x1 = make_branch(x[1], 64)
    x2 = make_branch(x[2], 128)
    x = fuse_layer_2([x0, x1, x2])

    # 2-3、中间层: 返回四个尺度的特征图
    x = transition_layer_3(x)
    x0 = make_branch(x[0], 32)  # [128, 128, 32]
    x1 = make_branch(x[1], 64)  # [64, 64, 64]
    x2 = make_branch(x[2], 128)  # [32, 32, 128]
    x3 = make_branch(x[3], 256)  # [16, 16, 256]
    x = fuse_layer_3([x0, x1, x2, x3])

    out = layers.Conv2D(filters=joint_nums, kernel_size=1)(x)
    model = keras.Model(inputs=inputs, outputs=out, name='HRNet')
    return model


if __name__ == '__main__':
    model_ = seg_hrnet((256, 256, 3), 17)
    model_.summary()
