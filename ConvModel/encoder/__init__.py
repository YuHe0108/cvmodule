"""常见的模型用于提取图像特征: vgg16、resnet50、mobilenet"""
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow import keras


def vanilla_encoder(input_height=224, input_width=224):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, 3))
    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(filter_size, (kernel, kernel),
                 padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel),
                padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad)))(x)
        x = (Conv2D(256, (kernel, kernel),
                     padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
                          ))(x)
        levels.append(x)

    return img_input, levels


def segnet_decoder(f, n_classes, n_up=3):
    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up - 2):
        o = (UpSampling2D((2, 2)))(o)
        o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
                    ))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               )(o)

    return o


def _segnet(n_classes, encoder, input_height=416, input_width=608,
            encoder_level=3):
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=3)
    model = Model(img_input, o, name='segnet')
    return model


def segnet(n_classes, input_height=416, input_width=608, encoder_level=3):
    model = _segnet(n_classes, vanilla_encoder, input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "segnet"
    return model


if __name__ == '__main__':
    segnet_ = segnet(1, 256, 256)
    segnet_.summary()
