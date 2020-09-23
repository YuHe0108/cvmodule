import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf


# --------------------------------------------------卷积单元层---------------------------------
def conv_block(inputs, filters, kernel_size, strides=1, relu=True):
    encoder = layers.Conv2D(
        filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    if relu:
        encoder = layers.ReLU()(encoder)
    else:
        encoder = layers.ELU()(encoder)
    return encoder


def encoder_block(inputs, filters, pool=True, relu=True):
    """下采样阶段"""
    encoder = conv_block(inputs, filters, kernel_size=3, relu=relu)
    encoder = conv_block(encoder, filters, kernel_size=3, relu=relu)
    if pool:
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(encoder)
        return encoder_pool, encoder
    else:
        return encoder


def decoder_block(input_tensor, concat_tensor, filters, relu=True):
    """上采样阶段: 对input_tensor上采样，之后与concat_tensor在通道方向上拼接"""
    decoder = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = conv_block(decoder, filters, kernel_size=3, relu=relu)
    decoder = conv_block(decoder, filters, kernel_size=3, relu=relu)
    return decoder


# --------------------------------------构建UNet模型--------------------------------------
def unet_model(input_shape, num_classes, dim=32, relu=True):
    inputs = keras.Input(shape=input_shape)
    # inputs_1 = conv_block(inputs, DIM, kernel_size=1, relu=relu)
    # 编码层， encoder0 经过卷积但是图像尺寸没变， encoder1缩小一半
    # encoder0_pool, encoder0 = encoder_block(inputs_1, DIM, relu=relu)  # 1
    encoder1_pool, encoder1 = encoder_block(inputs, dim, relu=relu)  # 1/2
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, dim * 2, relu=relu)  # 1/4
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, dim * 4, relu=relu)  # 1/8
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, dim * 8, relu=relu)  # 1/16

    # 中间层
    center = encoder_block(encoder4_pool, dim * 16, pool=False)  # 原始图像的1/16

    # 解码层
    decoder4 = decoder_block(center, encoder4, dim * 8, relu=relu)
    decoder3 = decoder_block(decoder4, encoder3, dim * 4, relu=relu)
    decoder2 = decoder_block(decoder3, encoder2, dim * 2, relu=relu)
    decoder1 = decoder_block(decoder2, encoder1, dim * 1, relu=relu)

    outputs = layers.Conv2D(num_classes, kernel_size=1, padding='same', strides=1)(decoder1)
    if num_classes == 1:
        outputs = layers.Activation('sigmoid')(outputs)
    else:
        outputs = layers.Softmax()(outputs)

    model = keras.Model(inputs, outputs, name='UNet')
    return model


if __name__ == '__main__':
    unet = unet_model(input_shape=(256, 256, 1), num_classes=1, dim=64)
    unet.summary()
