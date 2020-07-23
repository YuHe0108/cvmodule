# from tensorflow.keras import Model, Input
# from tensorflow.keras.applications import vgg16, resnet50
# from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Cropping2D, add, Dropout, Reshape, Activation)
# from tensorflow.keras import layers
# import tensorflow as tf
#
# """
# FCN-8特点:
#     1、不含全连接层(fc)的全卷积(fully conv)网络。可适应任意尺寸输入。
#     2、增大数据尺寸的反卷积(deconv)层。能够输出精细的结果。
#     3、结合不同深度层结果的跳级(skip)结构。同时确保鲁棒性和精确性。
#     4、使用 skip 结构融合多层（3层）输出，底层网络可以预测更多的位置信息，因为感受野小可以看到小的 pixels
#        上采样 lower-resolution layers 时，如果采样后的图因为 padding 等原因和前面的图大小不同，使用 crop,
#        当裁剪成大小相同的，spatially aligned ，使用 concat 操作融合两个层。
#
# FCN-8、FCN-16、FCN-32的区别与联系: 最后上采样的过程中，放大的倍数，
#     1、区别: FCN模型会输出三种尺寸的特征图: [b, 16, 16, filters], 这时候直接上采样32倍，可以得到 [b, 16*32, 16*32, n_classes],
#        如果直接上采样 32 倍预测输出，被称为 FCN-32。
#        FCN-16 和 FCN-8 则是融合了不同阶段的特征图，最终输出的时候，上采样16倍和8倍得到。
# """
#
#
# def fcn8_helper(input_shape, num_classes, backbone):
#     assert input_shape[0] % 32 == 0
#     assert input_shape[1] % 32 == 0
#
#     inputs = Input(input_shape)
#     if backbone == 'vgg16':
#         base_model = vgg16.VGG16(input_tensor=inputs,
#                                  include_top=False,
#                                  weights='imagenet',
#                                  pooling=None,
#                                  classes=100)
#     elif backbone == 'resnet50':
#         base_model = resnet50.ResNet50(input_tensor=inputs,
#                                        include_top=False,
#                                        weights='imagenet',
#                                        pooling=None,
#                                        classes=1000)
#     assert isinstance(base_model, Model)
#     base_model.trainable = False  # 是否固定特征提取单元
#
#     out = Conv2D(
#         filters=1024, kernel_size=7, padding="same", activation="relu", name="fc6")(base_model.output)
#     out = Dropout(rate=0.5)(out)
#     out = Conv2D(
#         filters=1024, kernel_size=1, padding="same", activation="relu", name="fc7")(out)
#     out = Dropout(rate=0.5)(out)
#     out = Conv2D(
#         filters=num_classes, kernel_size=(1, 1), padding="same", activation="relu",
#         kernel_initializer="he_normal", name="score_fr")(out)
#
#     # [B, 8, 8, filters] * 2 --> [None, 16, 16, n_classes]
#     out = Conv2DTranspose(
#         filters=num_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score2")(out)
#
#     fcn8 = Model(inputs=inputs, outputs=out)
#     return fcn8
#
#
# def fcn8_model(input_shape, num_classes):
#     fcn8 = fcn8_helper(input_shape, num_classes, backbone='vgg16')
#
#     # "block4_pool" shape: [B, 16, 16, 512] 跳跃连接融合低级特征:
#     skip_con1 = Conv2D(
#         num_classes, kernel_size=(1, 1), padding="same", activation=None,
#         kernel_initializer="he_normal", name="score_pool4")(fcn8.get_layer("block4_pool").output)
#     Summed = add(inputs=[skip_con1, fcn8.output])
#
#     # [B, 32, 32, num_classes]
#     x = Conv2DTranspose(
#         num_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score4")(Summed)
#
#     # block3_pool: [B, 32, 32, filters]
#     skip_con2 = Conv2D(
#         num_classes, kernel_size=(1, 1), padding="same", activation=None,
#         kernel_initializer="he_normal", name="score_pool3")(fcn8.get_layer("block3_pool").output)
#     Summed2 = add(inputs=[skip_con2, x])
#
#     # 上采样8倍, 直接由 [B, 32, 32, filters] --> [B, 32*8, 32*8, n_classes]
#     outputs = Conv2DTranspose(
#         num_classes, kernel_size=(8, 8), strides=(8, 8), padding="valid",
#         activation='sigmoid', name="upsample")(Summed2)
#
#     if num_classes == 1:
#         outputs = layers.Activation('sigmoid')(outputs)
#     else:
#         outputs = layers.Softmax()(outputs)
#
#     fcn_model = Model(inputs=fcn8.input, outputs=outputs, name='FCN8s')
#     return fcn_model
#
#
# def fcn8_model_resnet50(input_shape, num_classes):
#     fcn8 = fcn8_helper(input_shape, num_classes, backbone='resnet50')
#
#     # "block4_pool" shape: [B, 16, 16, 1024] 跳跃连接融合低级特征:
#     skip_con1 = Conv2D(
#         num_classes, kernel_size=(1, 1), padding="same", activation=None,
#         kernel_initializer="he_normal", name="score_pool4")(fcn8.get_layer("conv4_block6_out").output)
#     Summed = add(inputs=[skip_con1, fcn8.output])
#
#     # [B, 32, 32, num_classes]
#     x = Conv2DTranspose(
#         num_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score4")(Summed)
#
#     # block3_pool: [B, 32, 32, 512]
#     skip_con2 = Conv2D(
#         num_classes, kernel_size=(1, 1), padding="same", activation=None,
#         kernel_initializer="he_normal", name="score_pool3")(fcn8.get_layer("conv3_block4_out").output)
#     Summed2 = add(inputs=[skip_con2, x])
#
#     # 上采样8倍, 直接由 [B, 32, 32, filters] --> [B, 32*8, 32*8, n_classes]
#     outputs = Conv2DTranspose(
#         num_classes, kernel_size=(8, 8), strides=(8, 8), padding="valid",
#         activation='sigmoid', name="upsample")(Summed2)
#
#     if num_classes == 1:
#         outputs = layers.Activation('sigmoid')(outputs)
#     else:
#         outputs = layers.Softmax()(outputs)
#
#     fcn_model = Model(inputs=fcn8.input, outputs=outputs, name='FCN8s')
#     return fcn_model
#
#
# if __name__ == '__main__':
#     # m = FCN8(15, 320, 320)
#     # from keras.utils import plot_model
#     #
#     # plot_model(m, show_shapes=True, to_file='model_fcn8.png')
#     # print(len(m.layers))
#     model_1 = fcn8_model_resnet50(input_shape=(256, 256, 3), num_classes=1)
#     model_1.summary()
#     # inputs = tf.keras.Input((256, 256, 3))
#     # base_model = resnet50.ResNet50(input_tensor=inputs,
#     #                                include_top=False,
#     #                                weights='imagenet',
#     #                                pooling=None,
#     #                                classes=1000)
#     # base_model.summary()
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Cropping2D, add, Dropout, Reshape, Activation)
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras import Model, Input
from tensorflow.keras import layers

"""
FCN-8特点:
    1、不含全连接层(fc)的全卷积(fully conv)网络。可适应任意尺寸输入。 
    2、增大数据尺寸的反卷积(deconv)层。能够输出精细的结果。 
    3、结合不同深度层结果的跳级(skip)结构。同时确保鲁棒性和精确性。
    4、使用 skip 结构融合多层（3层）输出，底层网络可以预测更多的位置信息，因为感受野小可以看到小的 pixels
       上采样 lower-resolution layers 时，如果采样后的图因为 padding 等原因和前面的图大小不同，使用 crop,
       当裁剪成大小相同的，spatially aligned ，使用 concat 操作融合两个层。

FCN-8、FCN-16、FCN-32的区别与联系: 最后上采样的过程中，放大的倍数，
    1、区别: FCN模型会输出三种尺寸的特征图: [b, 16, 16, filters], 这时候直接上采样32倍，可以得到 [b, 16*32, 16*32, n_classes],
       如果直接上采样 32 倍预测输出，被称为 FCN-32。
       FCN-16 和 FCN-8 则是融合了不同阶段的特征图，最终输出的时候，上采样16倍和8倍得到。
"""


def fcn8_helper(input_shape, num_classes, weight_name='imagenet'):
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    inputs = Input(input_shape)
    base_model = vgg16.VGG16(input_tensor=inputs,
                             include_top=False,
                             weights=weight_name,
                             pooling=None,
                             classes=100)
    assert isinstance(base_model, Model)
    # base_model.trainable = False  # 是否固定特征提取单元

    out = Conv2D(
        filters=1024, kernel_size=7, padding="same", activation="relu", name="fc6")(base_model.output)
    out = Dropout(rate=0.5)(out)
    out = Conv2D(
        filters=1024, kernel_size=1, padding="same", activation="relu", name="fc7")(out)
    out = Dropout(rate=0.5)(out)
    out = Conv2D(
        filters=num_classes, kernel_size=(1, 1), padding="same", activation="relu",
        kernel_initializer="he_normal", name="score_fr")(out)

    # [B, 8, 8, filters] * 2 --> [None, 16, 16, n_classes]
    out = Conv2DTranspose(
        filters=num_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score2")(out)

    fcn8 = Model(inputs=inputs, outputs=out)
    return fcn8


def fcn8_model(input_shape, num_classes):
    fcn8 = fcn8_helper(input_shape, num_classes)

    # "block4_pool" shape: [B, 16, 16, 512] 跳跃连接融合低级特征:
    skip_con1 = Conv2D(
        num_classes, kernel_size=(1, 1), padding="same", activation=None,
        kernel_initializer="he_normal", name="score_pool4")(fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    # [B, 32, 32, num_classes]
    x = Conv2DTranspose(
        num_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None, name="score4")(Summed)
    # block3_pool: [B, 32, 32, filters]
    skip_con2 = Conv2D(
        num_classes, kernel_size=(1, 1), padding="same", activation=None,
        kernel_initializer="he_normal", name="score_pool3")(fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])
    # 上采样8倍, 直接由 [B, 32, 32, filters] --> [B, 32*8, 32*8, n_classes]
    outputs = Conv2DTranspose(
        num_classes, kernel_size=(8, 8), strides=(8, 8), padding="valid",
        activation='sigmoid', name="upsample")(Summed2)

    if num_classes == 1:
        outputs = layers.Activation('sigmoid')(outputs)
    else:
        outputs = layers.Softmax()(outputs)

    fcn_model = Model(inputs=fcn8.input, outputs=outputs, name='FCN8s')
    # for layer_ in fcn_model.layers[:]:
    #     layer_.trainable = True
    return fcn_model


if __name__ == '__main__':
    # m = FCN8(15, 320, 320)
    # from keras.utils import plot_model
    #
    # plot_model(m, show_shapes=True, to_file='model_fcn8.png')
    # print(len(m.layers))
    model_1 = fcn8_model(input_shape=(256, 256, 3), num_classes=1)
    model_1.summary()
