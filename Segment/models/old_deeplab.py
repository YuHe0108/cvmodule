from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os


def sep_layer(inputs, filters, prefix, stride=1, k_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ 自定义的深度可分离卷积: sep_conv(BN+relu) + conv(BN+relu)
    prefix: 层名称的前缀
    k_size: depthwise卷积核的大小
    rate: 扩张卷积率
    depth_activation: depthwise卷积后是否采用激活函数
    epsilon: BN层使用
    """
    if stride == 1:
        depth_padding = 'same'
    else:
        # 不直接采用 padding = 'same'的方式，而是采用ZeroPadding
        k_size_effective = k_size + (k_size - 1) * (rate - 1)
        pad_total = k_size_effective - 1  # 如果pad_total = 3-1=2
        pad_beg = pad_total // 2  # 2//2=1
        pad_end = pad_total - pad_beg  # 2-1 =1
        inputs = layers.ZeroPadding2D((pad_beg, pad_end))(inputs)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.ReLU()(inputs)

    # depth_wise
    x = layers.DepthwiseConv2D(k_size, strides=stride, dilation_rate=rate, padding=depth_padding,
                               name=f"{prefix}_{'depthwise'}", use_bias=False)(x)
    x = layers.BatchNormalization(name=f"{prefix}_{'depthwise_BN'}", epsilon=epsilon)(x)
    if depth_activation:
        x = layers.ReLU()(x)

    # 整合特征图 conv 1x1, strides=1
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False, name=f"{prefix}_{'pointwise'}")(x)
    x = layers.BatchNormalization(name=f"{prefix}_{'pointwise_BN'}", epsilon=epsilon)(x)
    if depth_activation:
        x = layers.ReLU()(x)
    return x


def conv_same_layer(inputs, filters, prefix, k_size=3, stride=1, rate=1):
    """
    不该变特征图的大小的卷积层 (当rate>1的时候，为扩张卷积)
    """
    if stride == 1:
        return layers.Conv2D(filters, k_size, stride, padding='same',
                             use_bias=False, dilation_rate=rate, name=prefix)(inputs)
    else:
        k_size_effective = k_size + (k_size - 1) * (rate - 1)
        pad_total = k_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = layers.ZeroPadding2D((pad_beg, pad_end))(inputs)
        return layers.Conv2D(filters, k_size, stride, padding='valid',
                             use_bias=False, dilation_rate=rate, name=prefix)(inputs)


def xception_block(inputs, depth_list, prefix, skip_connect_type, stride, rate=1,
                   depth_activation=False, return_skip=False):
    """用于构建xception，同样用到了残差结构，但是将卷积换成了 深度可分离卷积（depthwise + pointwise + conv 1x1）"""
    residual = inputs
    for i in range(3):
        # depthwise + pointwise + conv2d
        residual = sep_layer(residual, depth_list[i], prefix + '_separable_conv{}'.format(i + 1),
                             stride=stride if stride == 2 else 1, rate=rate, depth_activation=depth_activation)
        if i == 1:
            skip = residual  # 两次: depth_wise + conv2d

    # inputs: (None, 128, 128, 64) residual: (None, 16, 16, 128) skip: (None, 32, 32, 128)
    if skip_connect_type == 'conv':
        # 采用跳跃连接: 输入经过侧边conv后与主路输出相加
        shortcut = conv_same_layer(inputs, depth_list[-1], prefix + '_shortcut', k_size=1, stride=stride)
        shortcut = layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        output = layers.Add()([residual, shortcut])
    elif skip_connect_type == 'sum':
        # 采用跳跃连接直接与输入相加
        output = layers.Add()([residual, shortcut])
    elif skip_connect_type == 'none':
        # 不采用跳跃连接
        output = residual

    if return_skip:
        # output是整个block的输出，skip只是主路的经过两次sep_conv的输出
        return output, skip
    else:
        return output


def make_divisible(v, divisor, min_value=None):
    # 如果v=100， divisor=10
    if min_value is None:
        min_value = divisor  # min_value = 10
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # new_v = max(10, (100+5)//10*10=100) = 100
    if new_v < 0.9 * v:  # 100 < 0.9 * 100 (90)
        new_v += divisor
    return new_v


def inverse_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    """倒置的残差结构，中间特征图数量多于两头，用于构建mobile-net_v2, 并且将中间卷积层替换为了depthwise_conv,
    可以用 skip_connection 选择是否是残差结构, 最后的输出没有激活函数
    """
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)  # 如果=128,确保pointwise_filters是8的倍数
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)  # 那么pointwise_filters=128

    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # 先增加卷积层特征图的通道数
        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                          use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = layers.Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # 再使用: Depthwise + BN + relu_6
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                               use_bias=False, padding='same', dilation_rate=rate,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False,
                      activation=None, name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def deeplab_v3(input_shape=(512, 512, 3), num_classes=21, backbone='xception',
               OS=16, alpha=1., weight_name='pascal_voc', pre_train=False):
    # 同时实现了以 mobile_net_v2 和 xception 为backbone的模型
    if backbone not in {'xception', 'mobilenetv2'}:
        raise ValueError("only 'xception`  or 'mobilenetv2' ")

    inputs = layers.Input(input_shape)
    if backbone == 'xception':
        if OS == 8:
            # os=8的效果要好于os=16的
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        # 下采样2倍
        x = layers.Conv2D(32, 3, strides=2, name='entry_flow_conv1_1', use_bias=False, padding='same')(inputs)
        x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = layers.ReLU()(x)

        x = conv_same_layer(x, 64, 'entry_flow_conv1_2', k_size=3, stride=1)
        x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = layers.ReLU()(x)

        # 一共三个xception——block
        x = xception_block(x, [128, 128, 128], 'entry_flow_block1', skip_connect_type='conv',
                           stride=2, depth_activation=False)
        x, skip_1 = xception_block(x, [256, 256, 256], 'entry_flow_block2', skip_connect_type='conv',
                                   stride=2, depth_activation=False, return_skip=True)
        x = xception_block(x, [728, 728, 728], 'entry_flow_block3', skip_connect_type='conv',
                           stride=entry_block3_stride, depth_activation=False)

        # 中间层
        for i in range(16):
            x = xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                               skip_connect_type='sum', stride=1, rate=middle_block_rate, depth_activation=False)

        # 输出层
        x = xception_block(x, [728, 1024, 1024], 'exit_flow_block1', skip_connect_type='conv', stride=1,
                           rate=exit_block_rates[0], depth_activation=False)
        x = xception_block(x, [1536, 1536, 2048], 'exit_flow_block2', skip_connect_type='none', stride=1,
                           rate=exit_block_rates[0], depth_activation=True)
    else:
        first_block_filters = make_divisible(32 * alpha, 8)  # 32
        # down sample 2x
        x = layers.Conv2D(first_block_filters, kernel_size=3, strides=2, padding='same',
                          use_bias=False, name='Conv')(inputs)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = layers.Activation(tf.nn.relu6, name='Conv_relu6')(x)

        # inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate = 1
        # only depthwise_conv(BN+relu6) + conv (BN)
        x = inverse_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

        # downsample: 4x
        x = inverse_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
        x = inverse_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

        # downsample: 8x
        x = inverse_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
        x = inverse_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
        x = inverse_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

        # 不采用strides=2下采样，而采用扩张卷积的方式，增大感受野的面积, rate=2
        x = inverse_res_block(x, filters=64, alpha=alpha, stride=1, rate=1,
                              expansion=6, block_id=6, skip_connection=False)
        x = inverse_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=7, skip_connection=True)
        x = inverse_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=8, skip_connection=True)
        x = inverse_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=9, skip_connection=True)

        # strides=1, rate = 2
        x = inverse_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=10, skip_connection=False)
        x = inverse_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=11, skip_connection=True)
        x = inverse_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=12, skip_connection=True)

        # strides=1, rate = 2,4
        x = inverse_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,
                              expansion=6, block_id=13, skip_connection=False)
        x = inverse_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                              expansion=6, block_id=14, skip_connection=True)
        x = inverse_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                              expansion=6, block_id=15, skip_connection=True)

        # 最后rate=4
        x = inverse_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                              expansion=6, block_id=16, skip_connection=False)

    # 特征提取结束
    # 下面是: Atrous Spatial Pyramid Pooling
    b4 = layers.GlobalAveragePooling2D()(x)  # [b_size, channels]
    # (b_size, channels)->(b_size, 1, 1, channels)
    b4 = layers.Lambda(lambda x_: K.expand_dims(x_, 1))(b4)
    b4 = layers.Lambda(lambda x_: K.expand_dims(x_, 1))(b4)

    # [k_size, 1, 1, 256]
    b4 = layers.Conv2D(256, 1, padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(epsilon=1e-5, name='image_pooling_BN')(b4)
    b4 = layers.ReLU()(b4)

    # Decode: 上采样, 使用resize恢复至 global average pool 之前特征图的大小
    size_before = K.int_shape(x)
    b4 = layers.Lambda(lambda x_:
                       tf.compat.v1.image.resize(x_, size_before[1:3], method='bilinear', align_corners=True))(b4)

    # conv 1x1
    b0 = layers.Conv2D(256, 1, padding='same', use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.ReLU()(b0)

    if backbone == 'xception':
        # rate = 6(12)
        b1 = sep_layer(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12(24)
        b2 = sep_layer(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18(36)
        b3 = sep_layer(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
        # concatenate ASPP branches & project
        x = layers.Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = layers.Concatenate()([b4, b0])

    x = layers.Conv2D(256, 1, padding='same', use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)

    if backbone == 'xception':
        x = layers.Lambda(lambda x_:
                          tf.compat.v1.image.resize(x_, skip_1.shape[1:3], method='bilinear', align_corners=True))(x)
        des_skip1 = layers.Conv2D(48, 1, padding='same', use_bias=False, name='feature_projection0')(skip_1)
        des_skip1 = layers.BatchNormalization(epsilon=1e-5, name='feature_projection0_BN')(des_skip1)
        des_skip1 = layers.ReLU()(des_skip1)

        x = layers.Concatenate()([x, des_skip1])

        x = sep_layer(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
        x = sep_layer(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    if (weight_name == 'pascal_voc' and num_classes == 21) or (weight_name == 'cityscapes' and num_classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = layers.Conv2D(num_classes, 1, padding='same', name=last_layer_name)(x)
    size_before3 = K.int_shape(inputs)
    x = layers.Lambda(lambda x_:
                      tf.compat.v1.image.resize(x_, size_before3[1:3], method='bilinear', align_corners=True))(x)

    # 最后输出
    if num_classes == 1:
        if num_classes == 1:
            x = layers.Activation('sigmoid', name='sigmoid')(x)
        else:
            x = layers.Activation('softmax', name='softmax')(x)
    model = keras.Model(inputs, x, name='deeplabv3plus')

    if pre_train:
        weight_dir = r'C:\Users\YingYing\.keras\models'
        if weight_name == 'pascal_voc':
            weight_path = os.path.join(
                weight_dir, 'deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5')
            model.load_weights(weight_path, by_name=True)
            for layer in model.layers[:-35]:
                layer.trainable = False
    return model


if __name__ == '__main__':
    model_ = deeplab_v3(input_shape=(256, 256, 1), num_classes=1, pre_train=False)
    model_.summary()
    # keras.utils.plot_model(model_, to_file='deeplabv3_mobilenet_v2.png', show_shapes=True, dpi=120)