import tensorflow as tf
import numpy as np
import os

from tf_package.Module.make_tfrecord import get_tfrecord_data


# 1.获取训练数据集
def get_data(batch_size, data_name=None, record_path=None, original_shape=(128, 128, 3),
             repeat=1, re_size=64, classes=10):
    """
    :param batch_size:
    :param data_name: tf-record文件的名称
    :param record_path: tf-record文件路径
    :param original_shape: tf-record在制作时保存图像的原始shape
    :param repeat:
    :param re_size: resize图像尺寸
    :param classes: tf-record保存了多少类图像
    :return:
    """
    if data_name == 'cifar10':
        return cifar_data(batch_size, repeat=repeat)
    elif data_name == 'face_128':
        assert record_path is not None
        record_path = os.path.join(record_path, data_name)
        assert os.path.exists(record_path)  # 判断文件是否存在

        return get_tfrecord_data(img_shape=original_shape, re_size=re_size,
                                 classes=classes, repeat=repeat, norm=True,
                                 tf_record_path=record_path,
                                 batch_size=batch_size)

    else:
        assert tf_record_path is not None
        record_path = os.path.join(tf_record_path, data_name)
        assert os.path.exists(record_path)  # 判断文件是否存在

        return get_tfrecord_data(img_shape=original_shape, re_size=re_size,
                                 classes=classes, repeat=repeat, norm=True,
                                 tf_record_path=record_path,
                                 batch_size=batch_size)


# 2.样本归一化
def data_pro(image, label):
    image = tf.cast(image, np.float32)
    image = image / 127.5 - 1  # 使所有的像素值归一到[-1, 1]之间
    return image, label


# 3.cifar10样本获取
def cifar_data(batch_size, concat=True, repeat=1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if concat:
        x_train = np.concatenate((x_train, x_test), 0)
        y_train = np.concatenate((y_train, y_test), 0)

        y_train = tf.keras.utils.to_categorical(y_train, 10)

        cifar = tf.data.Dataset.from_tensor_slices((x_train, y_train)). \
            shuffle(10000).batch(batch_size, drop_remainder=True).repeat(repeat)
        cifar = cifar.map(data_pro)

        return cifar
    else:
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        train_data = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(50000).batch(batch_size, drop_remainder=True).repeat(repeat)
        test_data = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat(1)
        return train_data, test_data
