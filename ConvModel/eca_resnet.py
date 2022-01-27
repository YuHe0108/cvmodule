from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from cvmodule.Module import eca


def eca_resnet(inputs, num_classes, layers_per_block, block_type='basic_block', k_size=None):
    """通过 ECA Layer 构建的 ResNet, 参数介绍:
        :layer_per_block: 一个 block 由多少个 残差单元 组成
        :block_type: block的类型: basic block 或者 bottleneck
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]

    out = eca.conv_bn_activate(inputs, filters=64, k_size=7, strides=2)
    out = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(out)  # 已经降采样了4倍

    if block_type == 'basic_block':
        expansion = 1
        for _ in range(layers_per_block[0]):
            out = eca.eca_basic_block(out, filters=64, k_size=int(k_size[0]), strides=1)

        out = eca.eca_basic_block(out, filters=128, k_size=int(k_size[1]), strides=2)
        for _ in range(1, layers_per_block[1]):
            out = eca.eca_basic_block(out, filters=128, k_size=int(k_size[1]), strides=1)

        out = eca.eca_basic_block(out, filters=256, k_size=int(k_size[2]), strides=2)
        for _ in range(1, layers_per_block[2]):
            out = eca.eca_basic_block(out, filters=256, k_size=int(k_size[2]), strides=1)

        out = eca.eca_basic_block(out, filters=512, k_size=int(k_size[3]), strides=2)
        for _ in range(1, layers_per_block[3]):
            out = eca.eca_basic_block(out, filters=512, k_size=int(k_size[3]), strides=1)
    else:
        expansion = 4
        for _ in range(layers_per_block[0]):
            out = eca.eca_bottleneck(out, filters=64, k_size=int(k_size[0]), strides=1)

        out = eca.eca_bottleneck(out, filters=128, k_size=int(k_size[1]), strides=2)
        for _ in range(1, layers_per_block[1]):
            out = eca.eca_bottleneck(out, filters=128, k_size=int(k_size[1]), strides=1)

        out = eca.eca_bottleneck(out, filters=256, k_size=int(k_size[2]), strides=2)
        for _ in range(1, layers_per_block[2]):
            out = eca.eca_bottleneck(out, filters=256, k_size=int(k_size[2]), strides=1)

        out = eca.eca_bottleneck(out, filters=512, k_size=int(k_size[3]), strides=2)
        for _ in range(1, layers_per_block[3]):
            out = eca.eca_bottleneck(out, filters=512, k_size=int(k_size[3]), strides=1)

    # 输出层
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.Dense(num_classes)(out)
    return out


def eca_resnet18(input_shape, num_classes, k_size=None):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification

    train_params:
        Total params: 11,699,148
        Trainable params: 11,689,548
        Non-trainable params: 9,600
        :param k_size:
        :param num_classes:
        :param input_shape:
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]
    inputs = keras.Input(input_shape)
    out = eca_resnet(inputs,
                     num_classes=num_classes,
                     layers_per_block=[2, 2, 2, 2],
                     block_type='basic_block',
                     k_size=k_size)
    return keras.Model(inputs, out, name='resnet18')


def eca_resnet34(input_shape, num_classes=1000, k_size=None):
    """Constructs a ResNet-34 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification

    train_params:
        Total params: 21,814,770
        Trainable params: 21,797,746
        Non-trainable params: 17,024
        :param input_shape:
        :param k_size:
        :param num_classes:
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]

    inputs = keras.Input(input_shape)
    out = eca_resnet(inputs,
                     num_classes=num_classes,
                     layers_per_block=[3, 4, 6, 3],
                     block_type='basic_block',
                     k_size=k_size)
    return keras.Model(inputs, out, name='eca_resnet34')


def eca_resnet50(input_shape, num_classes=1000, k_size=None):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification

    total_params:
        Total params: 25,610,238
        Trainable params: 25,557,118
        Non-trainable params: 53,120
        :param input_shape:
        :param k_size:
        :param num_classes:
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]
    inputs = keras.Input(input_shape)
    out = eca_resnet(inputs,
                     num_classes=num_classes,
                     layers_per_block=[3, 4, 6, 3],
                     block_type='bottle_neck',
                     k_size=k_size)
    return keras.Model(inputs, out, name='eca_resnet50')


def eca_resnet101(input_shape, num_classes=1000, k_size=None):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification

    total_params:
        Total params: 44,654,675
        Trainable params: 44,549,331
        Non-trainable params: 105,344
        :param input_shape:
        :param k_size:
        :param num_classes:
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]
    inputs = keras.Input(input_shape)
    out = eca_resnet(inputs,
                     num_classes=num_classes,
                     layers_per_block=[3, 4, 23, 3],
                     block_type='bottle_neck',
                     k_size=k_size)
    return keras.Model(inputs, out, name='eca_resnet101')


def eca_resnet152(input_shape, num_classes=1000, k_size=None):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification

    total_params:
        Total params: 60,344,488
        Trainable params: 60,193,064
        Non-trainable params: 151,424
        :param input_shape:
        :param k_size:
        :param num_classes:
    """
    if k_size is None:
        k_size = [3, 3, 3, 3]
    inputs = keras.Input(input_shape)
    out = eca_resnet(inputs,
                     num_classes=num_classes,
                     layers_per_block=[3, 8, 36, 3],
                     block_type='bottle_neck',
                     k_size=k_size)
    return keras.Model(inputs, out, name='eca_resnet152')


if __name__ == '__main__':
    model_ = eca_resnet101((256, 256, 3), 1000)
    model_.summary()
