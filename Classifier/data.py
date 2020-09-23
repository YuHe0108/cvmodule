import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import pathlib
import os

from tf_record import TF_Record
from tf_package import utils

# 数据的名称、尺寸、类别数量
classes = 0
record_ori_size = 0


# 1.获取训练数据集
def get_data(batch_size, re_size, data_name, tfrecord_path=None, repeat=1, ori_size=None, augment=False):
    # 可以获取不同分辨率的图像
    if str(data_name).lower() == 'cifar10':
        return cifar_data(batch_size, re_size)
    else:
        assert tfrecord_path is not None
        if ori_size is None:
            ori_size = (record_ori_size, record_ori_size, 1)
        return get_tfrecord_data(tfrecord_path, batch_size, repeat, re_size, ori_size=ori_size, augment=augment)


# 2.样本归一化
def data_pro(re_size):
    def __data_pro(image, label):
        image = tf.image.resize(image, (re_size, re_size))
        image = tf.cast(image, np.float32)
        image = (image / 255.0 - 0.5) * 2  # 使所有的像素值归一到[-1, 1]之间
        return image, label

    return __data_pro


# 3.cifar10样本获取
def cifar_data(batch_size, re_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = np.concatenate((x_train, x_test), 0)
    y_train = np.concatenate((y_train, y_test), 0)

    y_train = tf.keras.utils.to_categorical(y_train, classes)

    cifar = tf.data.Dataset.from_tensor_slices((x_train, y_train)). \
        shuffle(10000).batch(batch_size, drop_remainder=True).repeat(1)
    cifar = cifar.map(data_pro(re_size))

    return cifar


def get_tfrecord_data(tf_record_path, batch_size, repeat, re_size, ori_size, augment=False):
    tfrecord = TF_Record(image_shape=ori_size, zip_file=True)
    if os.path.isdir(tf_record_path):
        data_set = tfrecord.from_dir_get_data(batch_size, tf_record_path, repeat, augment=augment)
    else:
        data_set = tfrecord.data_iterator(batch_size, tf_record_path, repeat, augment=augment)
    data_set = data_set.map(data_pro(re_size))
    return data_set


if __name__ == '__main__':
    train_data = get_data(batch_size=100, re_size=32, data_name='cifar10',
                          tfrecord_path=r'J:\DATA\Brain\TF_Records\data')
    for image_, label_ in train_data.take(1):
        print(image_.shape)
        print(np.max(image_[0]))
        print(np.min(image_[0]))
        # image_ = tf.compat.v1.image.resize_nearest_neighbor(image_, (128, 128))
        # print(image_[0].shape)
        # print(image_[0])
        # image_ex = np.uint8((image_[0] / 2 + 0.5) * 255.0)
        # plt.imshow(np.squeeze(image_ex))
        # plt.show()
        real_image = utils.make_gird(image_, n_rows=10, denorm=True, padding=0)
        plt.imsave('sample_32.png', real_image)
