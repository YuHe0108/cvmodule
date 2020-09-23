import os
import math
import string
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers, models
import tensorflow.keras.utils as keras_utils
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
import functools
import numpy as np

WEIGHTS_DIR_ROOT = r'C:\Users\YingYing\.keras\models'

BlockArgs = collections.namedtuple(
    'BlockArgs', [
        'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
        'expand_ratio', 'id_skip', 'strides', 'se_ratio'
    ])

# 1.1 基础 efficient-net B0 的结构参数
DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]


def conv_kernel_initializer(shape, dtype=None):
    """卷积核初始化
    和 tf.variance_scaling_initializer最大不同之处就是在于，tf.variance_scaling_initializer 使用的是 truncated norm,
    但是却具有未校正的标准偏差，而这里使用正态分布。类似地，tf.initializers.variance_scaling使用带有校正后的标准偏差。
    Args:
      shape: 卷积核的shape
      dtype: 卷积核的dtype
    Returns:
      经过初始化后的卷积核
    """
    kernel_height, kernel_width, input_filters, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None):
    """ 对全连接层的初始化
    这种出池化的方式等于
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',distribution='uniform').

    Args:
      shape: shape of variable
      dtype: dtype of variable
    Returns:
      初始化后的权重矩阵
    """
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


# 1.2 卷积核的初始化
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {'scale': 2.0,
               'mode': 'fan_out',
               'distribution': 'normal'
               }
}
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {'scale': 1. / 3.,
               'mode': 'fan_out',
               'distribution': 'uniform'
               }
}


def get_swish(x):
    # 激活函数swish
    return tf.nn.swish(x)


def round_filters(filters, width_coefficient, depth_divisor):
    """计算通过channel系数的缩放后filter的数量, 保证了经过放缩后的系数是8的整数倍
    并且保证输出的 filters 要大于输入 filters 的 90% 以上
    """
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    # 通过深度缩放系数后每个stage重复叠加的数量, 向上取整
    return int(math.ceil(depth_coefficient * repeats))


def get_dropout():
    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix=''):
    """在mobile-net中使用的倒置residual block"""
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    Dropout = get_dropout()

    # 扩张channels
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        # 使用全局池化层将特征图缩小到只有: [batch_size, channels], 之后通过Reshape: [batch_size, 1, 1, channels]
        # 最后与输入的特征图相乘得到输出
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
        target_shape = (1, 1, filters)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)

        se_tensor = layers.Conv2D(num_reduced_filters, 1, activation=activation, padding='same',
                                  use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1, activation='sigmoid', padding='same', use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters, 1, padding='same',
                      use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)

    # 是否使用跳跃连接, 如果选择了跳跃连接，那么strides=1，并且输出的filters和输入的filters的数量一致
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def efficient_net(width_coefficient, depth_coefficient, default_resolution,
                  dropout_rate=0.2, drop_connect_rate=0.2, depth_divisor=8,
                  blocks_args='none', model_name='efficientnet',
                  include_top=True, weights='imagenet', input_tensor=None,
                  input_shape=None, pooling=None, classes=1000, **kwargs):
    if blocks_args == 'none':
        blocks_args = DEFAULT_BLOCKS_ARGS

    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    activation = get_swish

    # 主体部分: 第一层卷积层
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2), padding='same', use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient)
        )

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args, activation=activation,
                          drop_rate=drop_rate, prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            for bidx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total

                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(x, block_args, activation=activation, drop_rate=drop_rate, prefix=block_prefix)
                block_num += 1

    # Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
        else:
            pass

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # 创建模型
    model = models.Model(inputs, x, name=model_name)

    # 加载权重
    if weights == 'imagenet':
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        weights_path = os.path.join(WEIGHTS_DIR_ROOT, file_name)
        model.load_weights(weights_path)

    elif weights == 'noisy-student':
        if include_top:
            file_name = "{}_{}.h5".format(model_name, weights)
        else:
            file_name = "{}_{}_notop.h5".format(model_name, weights)
        weights_path = os.path.join(WEIGHTS_DIR_ROOT, file_name)
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def efficient_net_b0(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.0, 1.0, 224, 0.2, model_name='efficientnet-b0', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b1(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.0, 1.1, 240, 0.2, model_name='efficientnet-b1', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b2(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.1, 1.2, 260, 0.3, model_name='efficientnet-b2', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b3(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.2, 1.4, 300, 0.3, model_name='efficientnet-b3', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b4(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(1.4, 1.8, 380, 0.4, model_name='efficientnet-b4', include_top=include_top, weights=weights,
                         input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b5(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.6, 2.2, 456, 0.4, model_name='efficientnet-b5', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b6(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        1.8, 2.6, 528, 0.5, model_name='efficientnet-b6', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_b7(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        2.0, 3.1, 600, 0.5, model_name='efficientnet-b7', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def efficient_net_l2(include_top=True, weights='imagenet', input_tensor=None,
                     input_shape=None, pooling=None, classes=1000):
    return efficient_net(
        4.3, 5.3, 800, 0.5, model_name='efficientnet-l2', include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes)


def get_efficient_net(phi=0, **kwargs):
    efficient_nets = [efficient_net_b0, efficient_net_b1, efficient_net_b2, efficient_net_b3,
                      efficient_net_b4, efficient_net_b5, efficient_net_b6, efficient_net_b7,
                      efficient_net_l2, ]
    return efficient_nets[phi](**kwargs)


if __name__ == '__main__':
    model_0 = efficient_net_b0()
    model_0.summary()
    from keras.applications.imagenet_utils import decode_predictions
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 as cv

    image = np.array(plt.imread('panda.jpg'))
    image = cv.cvtColor(cv.resize(image, (224, 224)), cv.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 127.5 - 1
    image = np.expand_dims(image, 0)
    output = model_0.predict(image)
    print(decode_predictions(output))

"""
模型的dropout_rate, 随着深度的增加，rate逐渐增加
0.025
0.05
0.07500000000000001
0.08750000000000001
0.1125
0.125
0.15000000000000002
0.1625
0.17500000000000002
"""
