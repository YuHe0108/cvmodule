import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import pathlib
import time
import tqdm
import os

""" 将数据集制作为 tfrecord 的形式，可用于训练GAN和分类器模型"""


class TFRecord:
    def __init__(self, image_shape, zip_file=True):
        assert len(image_shape) == 3  # 输入的image_shape是三位数字
        self.image_shape = image_shape
        self.zip_file = zip_file
        self.gray_scale = False
        if image_shape[-1] == 1:
            self.gray_scale = True
        if self.zip_file:
            self.options = tf.io.TFRecordOptions(compression_type='GZIP')

    @staticmethod
    def get_path_and_label(data_dir, label=None, sample_nums=50000):
        """
        1.读取文件下所有的文件，并返回所有文件的绝对路径与文件夹名
        2.文件夹的名称为文件的标签信息
        3.文件目录的层级信息，在一个大的文件夹下，包含多个子类文件夹
        """
        all_images_paths = []
        all_folder_names = []  # 图像的上层文件名为图像的标签
        for sub_file in pathlib.Path(data_dir).iterdir():
            for path in pathlib.Path(sub_file).iterdir():
                all_images_paths.append(str(path))
                all_folder_names.append(path.parent.name)

        folder_name_set = sorted(set(all_folder_names))  # 统计label的种类
        folder_name_to_label = {folder_name: index for index, folder_name in enumerate(folder_name_set)}
        """
        可以通过下标寻找label的字符名称
        label_to_folder_name = {item[1]: item[0] for item in folder_name_to_label.items()}
        将所有图像的标签由文字转换为数字
        """
        if label is not None:
            assert len(folder_name_set) == 1  # 只有一个文件夹
            all_images_labels = [label] * len(all_images_paths)
            print('设置的标签为: {}'.format(label))
        else:
            all_images_labels = [folder_name_to_label.get(folder_name) for folder_name in all_folder_names]
            print('folder_name_to_label: {}'.format(folder_name_to_label))

        return all_images_paths[:sample_nums], all_images_labels[:sample_nums]

    def make_record(self, all_file_paths, all_file_labels, record_save_path, record_save_name):
        save_record_path = os.path.join(record_save_path, record_save_name)
        data_len = len(all_file_paths)
        data_zip = zip(all_file_paths, all_file_labels)

        if not os.path.exists(save_record_path):

            if self.zip_file:
                writer = tf.io.TFRecordWriter(save_record_path, self.options)
            else:
                writer = tf.io.TFRecordWriter(save_record_path)

            for path, label in tqdm.tqdm(data_zip, total=data_len):
                try:
                    image = cv.imread(path)
                    image = cv.resize(image, tuple(self.image_shape[:2]))
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    if len(image.shape) == 4:
                        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
                    if image is None:
                        print('无法读取图像: {}'.format(path))
                    if self.gray_scale:
                        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                        if image.shape != self.image_shape[:2]:
                            print('异常文件的大小: {}, 路径为: {}'.format(image.shape, path))
                            break
                    else:
                        if image.shape != self.image_shape:
                            print('异常文件的大小: {}, 路径为: {}'.format(image.shape, path))
                            break
                    image_raw = image.tobytes()
                    label = int(label)

                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                        }
                    ))
                    writer.write(example.SerializeToString())  # 序列化为字符串
                except Exception as e:
                    print(path, '错误:'.format(e))
            writer.close()
        print('制作TF-Records完成!')

    def _parse_function(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'img_raw': tf.io.FixedLenFeature([], tf.string)
            }
        )
        # 取出我们需要的数据（标签，图片）
        label = features['label']
        image = features['img_raw']
        image = tf.io.decode_raw(image, tf.uint8)
        image = tf.reshape(image, self.image_shape)
        return image, label

    def data_iterator(self, batch_size, tf_record_path, repeat=1):
        # 声明TFRecordDataset
        if self.zip_file:
            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(buffer_size=200).repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset

    def from_dir_get_data(self, batch_size, tf_dir, repeat=1):
        """
        读取一系列tfrecord文件作为一个训练集
        传入的tf_dir下保存了tf_records文件
        """
        tf_list = []
        for path in pathlib.Path(tf_dir).iterdir():
            tf_list.append(str(path))

        dataset = tf.data.Dataset.list_files(tf_list).repeat(1)
        if self.zip_file:
            dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
                cycle_length=tf.data.experimental.AUTOTUNE,
            )
        else:
            dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=tf.data.experimental.AUTOTUNE,
            )
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(buffer_size=100).repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset


# 2.样本归一化
def data_pro(re_size, classes=10):
    def _data_pro(image, label):
        label = tf.one_hot(label, classes)
        image = tf.compat.v1.image.resize_nearest_neighbor(image, (re_size, re_size))
        image = tf.cast(image, np.float32)
        image = (image / 255.0 - 0.5) * 2  # 使所有的像素值归一到[-1, 1]之间
        return image, label

    return _data_pro


# 获取数据集
def get_tfrecord_data(img_shape, re_size, tf_record_path, batch_size, classes=10, repeat=1, norm=True):
    tfcord = TFRecord(img_shape, zip_file=True)
    dataset = tfcord.data_iterator(batch_size, tf_record_path, repeat)
    if norm:
        dataset = dataset.map(data_pro(re_size, classes=classes))
    return dataset


if __name__ == '__main__':
    data_root = r'E:\DATA\Celeba\faces'
    tf_record = TFRecord(image_shape=(128, 128, 3))
    all_paths, all_labels = tf_record.get_path_and_label(data_root, label=0)
    tf_record.make_record(all_paths, all_labels,
                          record_save_path='', record_save_name='face_128')
    data_set = get_tfrecord_data(img_shape=(128, 128, 3), re_size=64,
                                 tf_record_path='..\\face_128', batch_size=32)
    for x, y in data_set:
        print(x.shape)
        print(y.shape)
        print(y)
        plt.imshow(x[0])
        plt.show()
        break
