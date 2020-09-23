from tensorflow.keras import initializers, regularizers, constraints
import tensorflow as tf


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i * self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out


class BottleNeck(tf.keras.Model):
    def __init__(self, in_channel, mid_channel, out_channel, strides, radix, groups, reduction_factor):
        super(BottleNeck, self).__init__()
        self.radix = radix
        self.groups = groups
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.inter_channels = max(in_channel * radix // reduction_factor, 32)
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv1 = GroupConv2D(input_channels=in_channel,
                                 output_channels=mid_channel,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 groups=groups * radix,
                                 use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv2 = GroupConv2D(input_channels=mid_channel,
                                 output_channels=mid_channel * radix,
                                 kernel_size=3,
                                 strides=1,
                                 padding='same',
                                 groups=groups * radix,
                                 use_bias=False)
        self.fc1 = GroupConv2D(input_channels=self.mid_channel,
                               output_channels=self.inter_channels,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               groups=self.groups,
                               use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.fc2 = GroupConv2D(input_channels=self.inter_channels,
                               output_channels=self.mid_channel * self.radix,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               groups=self.groups,
                               use_bias=False)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.bn4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv4 = tf.keras.layers.Conv2D(self.out_channel, kernel_size=1, strides=strides, padding='same',
                                            use_bias=False)
        if self.in_channel != self.out_channel:
            self.conv3 = tf.keras.layers.Conv2D(self.out_channel, kernel_size=1, strides=1, padding='same',
                                                use_bias=False)
        else:
            self.sub_sample = tf.keras.layers.MaxPooling2D(pool_size=2, strides=strides, padding='same')

    def call(self, inputs, training=None, **kwargs):

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        splited = tf.split(x, num_or_size_splits=self.radix, axis=-1)
        z = tf.reshape(self.global_pool(sum(splited)), [-1, 1, 1, self.mid_channel])

        z = self.fc1(z)
        z = self.bn3(z, training=training)
        z = tf.nn.relu(z)
        z = self.fc2(z)

        # 实现r-softmax
        z = tf.reshape(z, [-1, self.groups, self.radix, self.mid_channel // self.groups])
        z = tf.transpose(z, [0, 2, 1, 3])
        z = tf.reshape(z, [-1, self.radix, self.mid_channel])
        z = tf.keras.activations.softmax(z, axis=1)

        logits = [tf.expand_dims(m, axis=1) for m in tf.split(z, num_or_size_splits=self.radix, axis=1)]
        out = sum([a * b for a, b in zip(splited, logits)])
        out = tf.nn.relu(self.bn4(out))
        out = self.conv4(out)

        if self.in_channel != self.out_channel:
            shortcut = self.conv3(inputs)
        else:
            shortcut = self.sub_sample(inputs)

        output = out + shortcut
        return output
