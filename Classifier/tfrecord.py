from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import partial
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import shutil
import pathlib
import random
import time
import tqdm
import os


def make_error_file(error_path='read_error_file'):
    """将读取出现错误的图像单独放在一个文件夹下"""
    if not os.path.exists(error_path):
        os.mkdir(error_path)
    return


def get_path_and_name(data_dir, sample_nums=60000):
    """读取路径下所有的文件，并返回所有文件的绝对路径和文件名"""
    all_paths = []
    all_names = []
    for path in pathlib.Path(data_dir).iterdir():
        all_paths.append(str(path))
        all_names.append(path.name)
    return all_paths[:sample_nums], all_names[:sample_nums]


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

    @staticmethod
    def augment_fun(image):
        """数据增强"""
        prob = random.uniform(0, 1)
        if 0 <= prob <= 0.2:
            k = random.randint(1, 3)
            image = tf.image.rot90(image, k=int(k))
        elif 0.2 < prob < 0.4:
            image = tf.image.flip_left_right(image)
        elif 0.4 < prob < 0.6:
            image = tf.image.flip_up_down(image)
        else:
            image = image
        return image

    def _parse_function(self, example_proto, augment):
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
        if augment:
            image = self.augment_fun(image)
        return image, label

    def data_iterator(self, batch_size, tf_record_path, repeat=1, augment=False):
        # 声明TFRecordDataset
        if self.zip_file:
            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(partial(self._parse_function, augment=augment))
        dataset = dataset.shuffle(buffer_size=2000).repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset

    def from_dir_get_data(self, batch_size, tf_dir, repeat=1, augment=False):
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
        dataset = dataset.map(partial(self._parse_function, augment=augment))
        dataset = dataset.shuffle(buffer_size=2000).repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset


if __name__ == '__main__':
    data_dir_ = r''  # 图像存放位置，主文件夹下只能有一个文件夹，并且不能有中文路径
    data_label = 0  # 目标标签
    tf_cord = TFRecord(image_shape=(256, 256, 3), zip_file=True)
    all_paths, all_labels = tf_cord.get_path_and_label(data_dir=data_dir_,
                                                       label=data_label)
    train_paths, test_paths, train_labels, test_labels = train_test_split(all_paths, all_labels, test_size=0.1)
    print('train_nums: ', len(train_paths))
    print('test_nums', len(test_paths))
    tf_cord.make_record(train_paths, train_labels,
                        record_save_path='', record_save_name='train_NORMAL_3_256_3_{}'.format(len(train_paths)))
    tf_cord.make_record(test_paths, test_labels,
                        record_save_path='', record_save_name='test_NORMAL_3_256_3_{}'.format(len(test_paths)))

    # print(len(all_paths), len(all_labels))
    # print(all_paths[:3])
    # print(all_labels[:3])
    # for image, label in tf_cord.from_dir_get_data(16, 'test_data').take(1):
    #     print(image.shape)
    #     print(label.shape)
    #     print(label)

    # tf_cord.make_tfrecord(file_path, file_label, tfrecord_save_path)

    # dataset = tf_cord.data_iterator(batch_size=64, read_tf_path='cat_dog_face')
    # for index, data in enumerate(dataset):
    #     print(index)
    #     # plt.imshow(data[0][10])
    #     # plt.show()
    #     # break
