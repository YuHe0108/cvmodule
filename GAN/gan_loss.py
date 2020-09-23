import tensorflow as tf
import numpy as np


def get_gan_loss(loss_type, from_logits=False):
    if loss_type == 'gan':
        return gan_loss(from_logits=from_logits)
    elif loss_type == 'wgan':
        return wgan_loss_fn()
    elif loss_type == 'wgan_gp':
        # 返回wgan损失函数（d_loss, g_loss），与乘法的梯度值
        return wgan_loss_fn(), gradient_penalty
    elif loss_type == 'lsgan':
        return lsgan_losses_fn()
    else:
        return None


def gan_loss(from_logits=False):
    # 传统的使用交叉熵作为损失函数: binary cross entropy
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def d_loss_fn(d_real_output, d_fake_output):
        r_loss = bce(y_true=tf.ones_like(d_real_output), y_pred=d_real_output)
        f_loss = bce(y_true=tf.zeros_like(d_fake_output), y_pred=d_fake_output)
        return r_loss + f_loss

    def g_loss_fn(d_fake_output):
        f_loss = bce(y_true=tf.ones_like(d_fake_output), y_pred=d_fake_output)
        return f_loss

    return d_loss_fn, g_loss_fn


def gradient_penalty(D, real, fake):
    """
        wgan衡量两个分布之间的差异，并且使用梯度惩罚的方式
        使得判别器满足利普西茨条件
    """
    # 插值
    shape = [tf.shape(real)[0]] + [1] * (real.shape.ndims - 1)  # [batch_size, 1, 1, 1]
    alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
    inter_data = real + alpha * (fake - real)
    inter_data.set_shape(real.shape)

    # 计算梯度
    with tf.GradientTape() as tape:
        tape.watch(inter_data)
        inter_pred = D(inter_data, training=True)
    if len(D.outputs) > 1:
        inter_pred = inter_pred[0]
    inter_grad = tape.gradient(inter_pred, inter_data)

    # 梯度惩罚
    # 计算的是欧几里得范数: 所有元素平方加和之后再开方
    norm = tf.norm(tf.reshape(inter_grad, [tf.shape(inter_grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.) ** 2)
    return gp


def wgan_loss_fn():
    """
    仅仅使用wgan，不使用梯度惩罚，正是因为没有加入梯度惩罚，
    有可能会导致d_r_output越来越大。
    D:
        d_r_output的值越大越好, d_f_output越小越好
    G:
        d_f_output的值越大越好
    """

    def d_loss_fn(d_r_output, d_f_output):
        r_loss = -tf.math.reduce_mean(d_r_output)
        f_loss = tf.math.reduce_mean(d_f_output)
        return r_loss + f_loss

    def g_loss_fn(d_f_output):
        f_loss = -tf.math.reduce_mean(d_f_output)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def lsgan_losses_fn():
    mse = tf.keras.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn
