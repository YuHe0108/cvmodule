from tensorflow import keras
import tensorflow as tf


def joint_mse_loss(y_pred, y_true, true_weight):
    """
    损失函数想要表达的意思: 输出的特征图数量为关键点的数量，意味着输出的是每一个像素属于各个关键点的置信度
    """
    batch_size = y_pred.shape[0]
    num_of_joints = y_pred.shape[-1]  # 有多少个关键点
    y_pred = tf.reshape(y_pred, shape=(batch_size, -1, num_of_joints))  # 合并宽和高
    heatmap_pred_list = tf.split(value=y_pred,
                                 num_or_size_splits=num_of_joints,
                                 axis=-1)  # 拆分每一个关键点的特征图 [batch_size, -1, 1]
    y_true = tf.reshape(y_true, shape=(batch_size, -1, num_of_joints))
    heatmap_true_list = tf.split(value=y_true,  # y_true执行与y_pred相同的操作
                                 num_or_size_splits=num_of_joints,
                                 axis=-1)
    losses = []  # 计算每一个关键点的损失值，并累加求平均
    for i in range(num_of_joints):
        heatmap_pred = tf.squeeze(heatmap_pred_list[i])
        heatmap_true = tf.squeeze(heatmap_true_list[i])
        loss = 0.5 * tf.losses.mean_squared_error(y_pred=heatmap_pred * true_weight[:, i],
                                                  y_true=heatmap_true * true_weight[:, i])
        losses.append(loss)
    return tf.reduce_mean(loss)


class JointsMSELoss(object):
    def __init__(self):
        self.mse = tf.losses.MeanSquaredError()

    def __call__(self, y_pred, target, target_weight):
        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]
        pred = tf.reshape(tensor=y_pred, shape=(batch_size, -1, num_of_joints))
        heatmap_pred_list = tf.split(value=pred, num_or_size_splits=num_of_joints, axis=-1)
        gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_of_joints))
        heatmap_gt_list = tf.split(value=gt, num_or_size_splits=num_of_joints, axis=-1)
        loss = 0.0
        for i in range(num_of_joints):
            heatmap_pred = tf.squeeze(heatmap_pred_list[i])
            heatmap_gt = tf.squeeze(heatmap_gt_list[i])
            loss += 0.5 * self.mse(y_true=heatmap_pred * target_weight[:, i],
                                   y_pred=heatmap_gt * target_weight[:, i])
        return loss / num_of_joints
