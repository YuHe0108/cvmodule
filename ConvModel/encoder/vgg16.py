import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def get_encoder(input_shape, pretrained='imagenet'):
    """将vgg16作为图像特征提取器， 输出五种不同尺度的特征图"""
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0
    assert input_shape[2] == 3
    weight_path = r'C:\Users\YingYing\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    img_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x  # [b, 32, 32, 256]

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x  # [b, 16, 16, 512]

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    f5 = x  # [b, 8, 8, 512]

    if pretrained == 'imagenet':
        keras.Model(img_input, x).load_weights(weight_path)
    return img_input, [f1, f2, f3, f4, f5]


if __name__ == '__main__':
    import numpy as np

    img_input, [f1, f2, f3, f4, f5] = get_encoder((256, 256, 3))
    img_input_, [f1_, f2_, f3_, f4_, f5_] = get_encoder((256, 256, 3), pretrained=None)
