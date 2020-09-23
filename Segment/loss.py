import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K


def cs_loss(y_true, y_pred, total_category, from_logits=False):
    """交叉熵损失"""
    if total_category > 2:
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
    else:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=from_logits)


def dyn_binary_weighted_cs_loss(y_true, y_pred):
    """
    可以根据每个batch动态调整正负样本权重.
    计算方法: 统计正样本、负样本的数量，并比上总样本的数量，各自作为正负样本的权重系数.
    y_true = [1, 0, 1],
    y_pred = [0.6, 0.7, 0.2]: 表示了正样本的可能性,
    """
    # num_pred = sum([0, 0, 1]) + sum([1, 0, 1]) = 1 + 2 = 3
    num_pred = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) + K.sum(y_true)
    # 正样本的权重系数: sum([1, 0, 1]) / 3 = 2/ 3
    zero_weight = K.sum(y_true) / (num_pred + K.epsilon())
    # 负样本权重系数: sum([0, 0, 1]) / 3 = 1 / 3
    one_weight = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) / (num_pred + K.epsilon())

    # calculate the weight vector
    weights = y_true * one_weight + (1.0 - y_true) * zero_weight
    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
    # apply the weights
    loss = weights * bin_crossentropy

    return K.mean(loss)


def dice_loss(y_true, y_pred):
    def dice_coeff():
        smooth = 1
        intersection = keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
        union = keras.backend.sum(y_true, axis=[1, 2, 3]) + keras.backend.sum(y_pred, axis=[1, 2, 3])
        score = keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        return score

    loss = 1 - dice_coeff()
    return loss


def binary_weighted_cs_loss(y_true, y_pred, postive_weight, from_logits=False):
    """带有权重的交叉熵损失
    positive_weight: 惩罚类别为1的力度，weight的值越大，惩罚的力度越大
    """
    negative_weight = 1 - postive_weight
    bin_cs_loss = cs_loss(y_true, y_pred, 2, from_logits=from_logits)
    weights = y_true * postive_weight + (1. - y_true) * negative_weight
    weighted_bin_cs_loss = weights * bin_cs_loss
    return K.mean(weighted_bin_cs_loss)


def binary_focal_loss(y_true, y_pred, p_weight=0.75, n_weight=0.25, gamma=3., epsilon=1e-6):
    """适用于二分类问题的 focal loss
    y_pred: 经过sigmoid之后的输出值
    gamma: 调制系数的值,
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    """
    positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    losses = -p_weight * K.pow(1. - positive, gamma) * K.log(
        positive + epsilon) - n_weight * K.pow(negative, gamma) * K.log(1. - negative + epsilon)
    return K.mean(losses)


def focal_dice_loss(y_true, y_pred):
    return 10 * dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


if __name__ == '__main__':
    import numpy as np

    # y_true_ = np.array([1, 0, 1], dtype=np.float32)
    # y_pred_ = np.array([0, 0, 1], dtype=np.float32)
    y_true_ = np.random.normal(size=(3, 4, 5, 6))
    y_pred_ = np.random.normal(size=(3, 4, 5, 6))
    print(dice_loss(y_true_, y_pred_))
