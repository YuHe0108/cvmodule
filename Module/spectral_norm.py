import tensorflow as tf

"""在SN-GAN中用到，奇异值归一化"""


class SpectralNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.training = training
        self.eps = eps
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError("not a standard keras layer")
        super(SpectralNorm, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        # 取出layer的kernel、shape
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False, name='sn_v', dtype=tf.float32)
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False, name='sn_u', dtype=tf.float32)
        super(SpectralNorm, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v
        if self.training:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)
        self.layer.kernel.assign(self.w / sigma)  # sigma为self.w的最大奇异值

    def restore_weights(self):
        self.layer.kernel.assign(self.w)
