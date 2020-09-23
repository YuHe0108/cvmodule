import tensorflow as tf
import numpy as np


def conv1x1(input_, output_dim, k_init=tf.initializers.glorot_uniform()):
    k_size = 1
    strides = 1
    output = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=k_size, strides=strides,
                                    padding='same', kernel_initializer=k_init)(input_)
    return output


def matmul_layer(x_1, x_2, shape):
    # 用层的方式实现矩阵乘法
    return tf.keras.layers.Layer(tf.matmul, output_shape=shape)((x_1, x_2))


class SelfAttnModel(tf.keras.Model):

    def __init__(self, input_dims, squeeze_times=8, **kwargs):
        super(SelfAttnModel, self).__init__(**kwargs)
        self.atten = _Attention()
        self.query_conv = tf.keras.layers.Conv2D(filters=input_dims // squeeze_times, kernel_size=1)
        self.key_conv = tf.keras.layers.Conv2D(filters=input_dims // squeeze_times, kernel_size=1)
        self.value_conv = tf.keras.layers.Conv2D(filters=input_dims, kernel_size=1)

    def call(self, inputs):
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        return self.atten([q, k, v, inputs])


class _Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(_Attention, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.gamma = self.add_weight(
            self.name + '_gamma', shape=(), initializer=tf.initializers.Zeros,
        )

    def call(self, inputs):
        if len(inputs) != 4:
            raise Exception('an attention layer should have 4 inputs')

        query_tensor = inputs[0]
        key_tensor = inputs[1]
        value_tensor = inputs[2]
        origin_input = inputs[3]

        input_shape = tf.shape(query_tensor)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        proj_query = tf.reshape(query_tensor, (batch_size, height * width, -1))  # [b, h*w, c]
        proj_key = tf.transpose(
            tf.reshape(key_tensor, (batch_size, height * width, -1)), (0, 2, 1))  # [b, c, h*w]
        proj_value = tf.transpose(
            tf.reshape(value_tensor, (batch_size, height * width, -1)), (0, 2, 1))  # [b, c, h*w]

        energy = tf.matmul(proj_query, proj_key)  # [b, h*w, h*w]
        attention = tf.nn.softmax(energy)  # [b, c, h*w] * [b, h*w, h*w] = [b, c, h*w]
        out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))
        out = tf.reshape(
            tf.transpose(out, (0, 2, 1)), (batch_size, height, width, -1))  # [b, h, w, c]
        return tf.add(tf.multiply(out, self.gamma), origin_input), attention


def attention_layer(inputs, dims, squeeze_times=8, name=None):
    _atten_layer = SelfAttnModel(name=name, input_dims=dims, squeeze_times=squeeze_times)
    return _atten_layer(inputs)


def test_model(input_shape=(32, 32, 3)):
    inputs = tf.keras.layers.Input(input_shape)
    outputs, attention_map = attention_layer(inputs, 64)
    print(outputs.shape)
    print(attention_map.shape)

    '''
    inputs_1 = tf.keras.layers.Reshape((32 * 32, 3))(inputs)
    inputs_1 = tf.keras.layers.Permute((2, 1))(inputs_1)  # [b, 3, 32*32]

    inputs_2 = tf.keras.layers.Reshape((32 * 32, 3))(inputs)  # [b, 32 * 32, 3]

    outputs = tf.keras.layers.Lambda(tf.matmul, output_shape=(3, 3), arguments={'b': inputs_2})(inputs_1)
    '''
    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    model_test = test_model()
    model_test.summary()
    print(model_test.output.shape)
    # print(model_test.non_trainable_weights[0].name)
