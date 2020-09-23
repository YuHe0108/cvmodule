import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def get_encoder(input_shape, pretrained='imagenet', base_model_trained=True):
    output_names = [
        'conv1_conv', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'
    ]
    # inputs:           [256, 256, 3]
    # conv1_conv:       [128, 128, 64]
    # conv2_block3_out: [64, 64, 256]
    # conv3_block4_out: [32, 32, 512]
    # conv4_block6_out: [16, 16, 1024]
    # conv5_block3_out: [8, 8, 2048]
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,
                                                         include_top=False,
                                                         weights=pretrained)
    base_model.trainable = base_model_trained
    outputs = []
    for name in output_names:
        outputs.append(base_model.get_layer(name).output)
    return base_model.inputs, outputs


if __name__ == '__main__':
    import numpy as np

    get_encoder(input_shape=(256, 256, 3))
