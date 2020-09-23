import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from tf_package.ConvModel.encoder import vgg16, resnet50


def crop(o1, o2, i):
    """裁剪o1和o2至相同的尺寸，裁剪的宽和高取两个特征图的最小值"""
    o_shape2 = keras.Model(i, o2).output_shape
    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = keras.Model(i, o1).output_shape
    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = layers.Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = layers.Cropping2D(cropping=((0, 0), (0, cx)))(o2)
    if output_height1 > output_height2:
        o1 = layers.Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = layers.Cropping2D(cropping=((0, cy), (0, 0)))(o2)
    return o1, o2


def fcn_8(input_shape, num_classes, encoder=vgg16, pretrained='imagenet'):
    img_input, levels = encoder.get_encoder(input_shape, pretrained=pretrained)
    [f1, f2, f3, f4, f5] = levels
    o = f5

    o = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)
    o = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)

    # 上采样
    o = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o)
    o = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f4
    o2 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o2)
    o, o2 = crop(o, o2, img_input)
    o = layers.Add()([o, o2])

    # 上采样
    o = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f3
    o2 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o2)
    o2, o = crop(o2, o, img_input)
    o = layers.Add()([o2, o])

    # 上采样
    o = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8),
                               use_bias=False, padding='same')(o)
    if num_classes == 1:
        o = layers.Activation('sigmoid')(o)
    else:
        o = layers.Softmax()(o)
    model = keras.Model(img_input, o, name='fcn8s')
    return model


def fcn_32(input_shape, num_classes, encoder=vgg16, pretrained='imagenet'):
    img_input, levels = encoder.get_encoder(input_shape, pretrained=pretrained)
    [f1, f2, f3, f4, f5] = levels
    o = f5

    o = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)
    o = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)

    o = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o)
    o = layers.Conv2DTranspose(num_classes, kernel_size=64, strides=32, use_bias=False, padding='same')(o)

    if num_classes == 1:
        o = layers.Activation('sigmoid')(o)
    else:
        o = layers.Softmax()(o)
    model = keras.Model(img_input, o, name='fcn32s')
    return model


def fcn_8_vgg(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_8(input_shape, num_classes, vgg16, pretrained=pretrained)
    model.model_name = "fcn_8s_vgg"
    return model


def fcn_32_vgg(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_32(input_shape, num_classes, vgg16, pretrained=pretrained)
    model.model_name = "fcn_32s_vgg"
    return model


def fcn_8_resnet50(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_8(input_shape, num_classes, resnet50, pretrained=pretrained)
    model.model_name = "fcn_8s_resnet50"
    return model


def fcn_32_resnet50(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_32(input_shape, num_classes, resnet50, pretrained=pretrained)
    model.model_name = "fcn_32s_resnet50"
    return model


def fcn_8_mobilenet(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_8(input_shape, num_classes, mobilenet, pretrained='imagenet')
    model.model_name = "fcn_8s_mobilenet"
    return model


def fcn_32_mobilenet(input_shape, num_classes, pretrained='imagenet'):
    model = fcn_32(input_shape, num_classes, mobilenet, pretrained='imagenet')
    model.model_name = "fcn_32s_mobilenet"
    return model


if __name__ == '__main__':
    fcn_8_model = fcn_8_vgg((256, 256, 3), 1)
    fcn_8_model.summary()
    # m = fcn_32(101)
