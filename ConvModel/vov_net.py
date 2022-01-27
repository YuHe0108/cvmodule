import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from cvmodule.Module import get_norm
from collections import OrderedDict

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv_utils(inputs,
                  filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  norm_name='batch_norm',
                  **kwargs):
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    x = layers.Conv2D(
        filters=filters, kernel_size=1, strides=1, use_bias=False,
        kernel_initializer=keras.initializers.he_normal())(x)
    x = get_norm.get_norm_layer(norm_name, **kwargs)(x)
    x = layers.ReLU()(x)
    return x


def conv_utils(inputs,
               filters,
               kernel_size=3,
               strides=1,
               padding='same',
               norm_name='batch_norm',
               **kwargs):
    x = layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        use_bias=False, kernel_initializer=keras.initializers.he_normal())(inputs)
    x = get_norm.get_norm_layer(norm_name, **kwargs)(x)
    x = layers.ReLU()(x)
    return x


def h_sigmoid(inputs):
    """优于relu6的最高输出为6，那么除以6之后，
    就将输出限制在了0-1之间，类似于sigmoid的功能"""
    return tf.nn.relu6(inputs + 3.0) / 6.0


def ese_utils(inputs, redution=4):
    """effective SE: SE-Net中提出的特征整合单元"""
    x = layers.GlobalAveragePooling2D()(inputs)  # [b, h, w, c] --> [b, c]
    x = layers.Reshape([1, 1, x.shape[-1]])(x)
    x = layers.Conv2D(x.shape[-1], kernel_size=1)(x)
    return layers.Multiply()([inputs, x])


def vovnet_v2_osa_utils(inputs,
                        out_filters,
                        block_filters,
                        layer_per_block,
                        depthwise=False,
                        identity=False,
                        se=False):
    """
    :param inputs:
    :param out_filters: 最终osa单元输出的filters
    :param block_filters: 每一层卷积所用的卷积核的数量
    :param layer_per_block: 每一个osa有多少个 卷积层
    :param se: 是否使用 efficient se
    :param depthwise: 是否使用 depthwise + pointwise
    :param identity: 是否加入跳跃连接
    :return:
    """
    x = inputs
    total_outputs = [x]
    for i in range(layer_per_block):
        if depthwise:
            x = dw_conv_utils(x, block_filters)
        else:
            x = conv_utils(x, block_filters)
        total_outputs.append(x)
    concat_output = layers.Concatenate()(total_outputs)

    # 拼接特征图之后，使用1x1的卷积整合特征图
    output = conv_utils(concat_output, filters=out_filters, kernel_size=1)
    output = ese_utils(output)

    # 是否使用跳跃连接
    if identity:
        output = layers.Add()([inputs, output])
    return output


def vovnet_v2_osa_stage(inputs,
                        out_filters,
                        stage_filters,
                        block_per_stage,
                        layer_per_block,
                        stage_num,
                        depthwise=False,
                        se=False):
    """
    :param inputs:
    :param out_filters: 输出channel的数量
    :param stage_filters: 每一个block中每一层使用 filter 的数量
    :param block_per_stage: 一个stage中有多少个block
    :param layer_per_block: 一个block中由多少个layer组成
    :param stage_num: 第几个stage
    :param se: 是否使用ese
    :param depthwise: 使用使用depthwise
    :return:
    """
    x = inputs
    # 只有当stage_num 不等于2的时候，才加入maxpooling
    if stage_num != 2:
        x = layers.MaxPooling2D(2, 2, padding='same')(x)

    if block_per_stage != 1:
        se = False

    # 这一步没有添加 跳跃连接，因为x和输出的特征图数量不品配
    x = vovnet_v2_osa_utils(x, out_filters, stage_filters, layer_per_block, se=se, depthwise=depthwise)
    for i in range(block_per_stage - 1):
        # 只有在最后输出的时候才会添加 ese 模块
        if i != block_per_stage - 2:
            se = False
        x = vovnet_v2_osa_utils(x, out_filters, stage_filters, layer_per_block,
                                se=se, identity=True, depthwise=depthwise)

    return x


def vovnet_v2(input_shape, norm_name, model_name):
    """
    :param input_shape:
    :param norm_name: 使用哪一个 normlize 层: batchnorm、layernorm、instancenorm
    :param model_name: 选择使用哪一个模型的配置
    :param stage_conv_filters: 每一个stage中的 conv 层用的 filters
    :param stage_out_filters: 每一个stage输出时的filters
    :param block_per_stage: 一个stage中含有多少个block
    :param layer_per_block: 一个block中有多少layer
    :param depthwise: 是否使用depthwise
    :param se: 是否使用 efficient se
    :return:
    """
    model_config = _STAGE_SPECS[model_name]
    stem_filters = model_config["stem"]
    stage_conv_filters = model_config["stage_conv_ch"]
    stage_out_filters = model_config["stage_out_ch"]
    block_per_stage = model_config["block_per_stage"]
    layer_per_block = model_config["layer_per_block"]
    se = model_config["eSE"]
    depthwise = model_config["dw"]

    outputs = OrderedDict()  # 输出特征图存放在字典中
    inputs = layers.Input(input_shape)
    # 输入层 3 层： 下采样了四倍
    x = conv_utils(inputs, stem_filters[0], strides=2)
    if depthwise:
        x = dw_conv_utils(x, stem_filters[1], strides=1)
        x = dw_conv_utils(x, stem_filters[2], strides=2)
    else:
        x = conv_utils(x, stem_filters[1], strides=1)
        x = conv_utils(x, stem_filters[2], strides=2)
    current_stride = 4  # 已经下采样了4倍
    # 特征图下采样的倍数
    out_feature_strides = {"stem": current_stride, "stage2": current_stride}

    # 'stem': [64, 64, 128],
    # "stage_conv_ch": [128, 160, 192, 224],
    # "stage_out_ch": [256, 512, 768, 1024],

    for i in range(4):  # VOVNet所有的变体都只有4个stage
        x = vovnet_v2_osa_stage(x,
                                out_filters=stage_out_filters[i],
                                stage_filters=stage_conv_filters[i],
                                block_per_stage=block_per_stage[i],
                                layer_per_block=layer_per_block,
                                stage_num=i + 2,  # 2, 3, 4, 5
                                depthwise=depthwise,
                                se=se)
        stage_name = f'stage_{i + 2}'
        outputs[stage_name] = x
        if i != 0:
            # 因为当stage=2的时候，不会下采样
            out_feature_strides[stage_name] = current_stride = int(current_stride * 2)

    output = [x, outputs]
    return keras.Model(inputs, output, name='VOVNet-V2')


if __name__ == '__main__':
    vovnet_v2_model = vovnet_v2((256, 256, 3), 'batch_norm', 'V-19-slim-dw-eSE')
    vovnet_v2_model.summary()
