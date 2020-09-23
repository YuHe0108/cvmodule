from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import math

"""
Refernce: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

在ECA-Net中，计算一维卷积核的大小方式:
    gamma = 2
    b = 1
    n, h, w, c = 8, 128, 128, 128
    t = int(abs(math.log(c, 2) + b) / gamma)
    k = t if t % 2 else t + 1
"""


def kernel_initializer(shape, dtype=None):
    # shape: [k_size, k_size, in_channels, out_channels]
    n = shape[0] * shape[1] * shape[-1]
    return keras.backend.random_normal(shape, stddev=math.sqrt(2. / n), dtype=dtype)


def conv_bn_activate(inputs, filters, k_size=3, strides=1, activate='relu'):
    out = layers.Conv2D(filters,
                        kernel_size=k_size,
                        strides=strides,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=kernel_initializer)(inputs)
    out = layers.BatchNormalization()(out)
    if activate == 'lrelu':
        return layers.LeakyReLU(0.1)(out)
    elif activate == 'relu':
        return layers.ReLU()(out)
    elif activate is None:
        return out


def eca_layer(inputs, gamma=2, b=1):
    # 根据 特征图的参数设置不同的卷积核尺寸
    batch_size, h, w, c = inputs.shape
    t = int(abs(math.log(c, 2) + b) / gamma)  # 特征图的数量越大, t的值也越大
    k_size = t if t % 2 else t + 1  # 保证了k_size是一个奇数

    output = layers.GlobalAveragePooling2D()(inputs)  # [batch_size, channel]
    output = layers.Conv1D(filters=1,
                           kernel_size=k_size,
                           strides=1,
                           padding='same',
                           use_bias=False)(tf.expand_dims(output, -1))  # [batch_size, 256, 1]
    output = tf.expand_dims(tf.transpose(output, perm=[0, 2, 1]), 1)  # [batch_size, 1, 1, 256]
    output = layers.Activation('sigmoid')(output)
    output = layers.Multiply()([inputs, output])  # [batch_size, height, width, 256]
    return output


def eca_basic_block(inputs, filters, k_size=3, strides=1, expansion=1):
    """通过 ECA 构建 ECA-Block 是一个残差单元"""
    residual = inputs
    out = conv_bn_activate(inputs, filters=filters, k_size=k_size, strides=strides)
    out = conv_bn_activate(out, filters=filters, k_size=k_size, strides=1, activate=None)

    out = eca_layer(out)
    if strides != 1 or expansion * filters != inputs.shape[-1]:
        residual = conv_bn_activate(inputs,
                                    filters=filters,
                                    k_size=1,
                                    strides=strides,
                                    activate=None)
    out = layers.Add()([residual, out])
    out = layers.ReLU()(out)
    return out


def eca_bottleneck(inputs, filters, k_size=3, strides=1, expansion=4):
    """同样也是由 ECA-Module 构建的残差单元，但是不同之处在于:
       filters -> filters -> filters * 4
    """
    residual = inputs
    out = conv_bn_activate(inputs, filters=filters, k_size=1, strides=1)
    out = conv_bn_activate(out, filters=filters, k_size=k_size, strides=strides)
    out = conv_bn_activate(out, filters=filters * expansion, k_size=1, strides=1, activate=None)
    out = eca_layer(out)
    if strides != 1 or filters * expansion != inputs.shape[-1]:
        residual = conv_bn_activate(inputs,
                                    filters=filters * expansion,
                                    k_size=1,
                                    strides=strides,
                                    activate=None)
    out = layers.Add()([residual, out])
    out = layers.ReLU()(out)
    return out


# class eca_layer(nn.Module):
#     """PyTorch: Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, channel, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()  # torch.Size([32, 256, 3, 3])
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)  # torch.Size([32, 256, 1, 1])
#
#         # Two different branches of ECA module # torch.Size([32, 1, 256])
#         print(y.squeeze(-1).transpose(-1, -2).shape, 2)
#         # torch.Size([32, 1, 256])
#         print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape, 3)
#         # torch.Size([32, 256, 1, 1])
#         print(self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).shape, 4)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # torch.Size([32, 256, 1, 1])
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)


if __name__ == '__main__':
    pass
    # a = tf.random.normal((32, 3, 3, 256))
    # out = eca_layer(a)
    # print(out.shape)
    # eca_layer_ = eca_layer(256, 3)
    # b = torch.randn((32, 256, 3, 3))
    # print(b.shape)
    # eca_layer_.forward(b)
