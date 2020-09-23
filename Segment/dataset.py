import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import pathlib
import tqdm
import os

from sklearn.model_selection import train_test_split


def get_path_and_name(data_dir, sample_nums=60000):
    """读取路径下所有的文件，并返回所有文件的绝对路径和文件名"""
    all_paths = []
    all_names = []
    for path in pathlib.Path(data_dir).iterdir():
        all_paths.append(str(path))
        all_names.append(path.name)
    return all_paths[:sample_nums], all_names[:sample_nums]


class TF_Record:
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
    def get_path_and_label(data_dir, label=None, sample_nums=60000):
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

    def check_img_file(self, images_list, color_choices):
        assert isinstance(images_list, list)
        out_images_list = []
        for index, image in enumerate(images_list):
            if len(image.shape) == 4:
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            if image is None:
                print('无法读取图像: {}'.format(path))
                raise FileNotFoundError
            if color_choices[index]:
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                if image.shape != self.image_shape[:2]:
                    print('异常文件的大小: {}, 路径为: {}'.format(image.shape, path))
                    raise FileExistsError
            else:
                if image.shape != self.image_shape:
                    print('异常文件的大小: {}, 路径为: {}'.format(image.shape, path))
                    raise FileExistsError
            out_images_list.append(image)
        return out_images_list

    def read_image(self, image_path_list):
        assert isinstance(image_path_list, list)
        read_images = []
        for image_path in image_path_list:
            image = cv.imread(image_path)
            if image is None:
                print(image_path)
                image = plt.imread(image_path)
            else:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, tuple(self.image_shape[:2]))
            read_images.append(image)
        return read_images

    def make_record(self, image_paths, mask_paths, record_save_path, record_save_name):
        save_record_path = os.path.join(record_save_path, record_save_name)
        data_len = len(image_paths)
        data_zip = zip(image_paths, mask_paths)

        if not os.path.exists(save_record_path):

            if self.zip_file:
                writer = tf.io.TFRecordWriter(save_record_path, self.options)
            else:
                writer = tf.io.TFRecordWriter(save_record_path)

            for image_path, mask_path in tqdm.tqdm(data_zip, total=data_len):
                try:
                    # 图像、图像的分割图像
                    images_list = self.read_image([image_path, mask_path])
                    check_images_list = self.check_img_file(images_list, [False, True])
                    images_raw_list = [check_image.tobytes() for check_image in check_images_list]

                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images_raw_list[0]])),
                            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images_raw_list[1]])),
                        }
                    ))
                    writer.write(example.SerializeToString())  # 序列化为字符串
                except Exception as e:
                    print(image_path, '错误:'.format(e))
            writer.close()
        print('制作TF-Records完成!')

    def _parse_function(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'mask': tf.io.FixedLenFeature([], tf.string),
            }
        )
        # 取出我们需要的数据（标签，图片）
        image = tf.io.decode_raw(features['image'], tf.uint8)
        mask = tf.io.decode_raw(features['mask'], tf.uint8)

        # 调整图像的shape
        image = tf.reshape(image, self.image_shape)
        mask = tf.reshape(mask, self.image_shape[:2])
        return image, mask

    def data_iterator(self, batch_size, tf_record_path, repeat=1, buffer_size=1000, is_train_data=True):
        # 声明TFRecordDataset
        if self.zip_file:
            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(self._parse_function)
        if is_train_data:
            dataset = dataset.cache().shuffle(buffer_size=buffer_size).repeat(repeat).batch(batch_size,
                                                                                            drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset

    def from_dir_get_data(self, batch_size, tf_dir, repeat=1, buffer_size=1000):
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
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset


# 2.样本归一化
def norm_image(image, re_size):
    image = tf.compat.v1.image.resize_nearest_neighbor(image, (re_size, re_size))
    image = tf.cast(image, np.float32)
    image = image / 127.5 - 1  # 使所有的像素值归一到[-1, 1]之间
    return image


def data_pro(re_size, augment=True):
    def _data_pro(image, mask_image):
        if augment:
            if tf.random.uniform(()) > 0.4:
                image = tf.image.random_brightness(image, max_delta=50.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

                image = tf.image.flip_left_right(image)
                mask_image = tf.image.flip_left_right(mask_image)

        image = norm_image(image, re_size)

        if len(mask_image.shape) == 3:
            mask_image = tf.expand_dims(mask_image, -1)
        mask_image = tf.compat.v1.image.resize_nearest_neighbor(mask_image, (re_size, re_size))
        mask_image = tf.squeeze(mask_image, -1)
        mask_image = tf.cast(mask_image, tf.float32)
        mask_image -= 1  # mask_image原有的像素值: [1, 2, 3] -> [0, 1, 2]
        return image, mask_image

    return _data_pro


# 获取数据集
def get_tfrecord_data(img_shape, re_size, tf_record_path, batch_size,
                      repeat=1, buffer_size=1000, norm=True, is_train_data=True):
    tf_cord = TF_Record(img_shape, zip_file=True)
    dataset = tf_cord.data_iterator(batch_size, tf_record_path, repeat, buffer_size, is_train_data)
    if norm:
        dataset = dataset.map(data_pro(re_size, is_train_data))
    return dataset


if __name__ == '__main__':
    ''''''
    root_dir = r'J:\DATA\OxfordCat'
    image_paths_ = r'J:\DATA\OxfordCat\images\images'
    mask_paths_ = r'J:\DATA\OxfordCat\annotations\annotations\trimaps'
    all_image_paths_, _ = get_path_and_name(image_paths_)
    all_mask_paths_, _ = get_path_and_name(mask_paths_)

    image_train, image_test, mask_train, mask_test = train_test_split(all_image_paths_,
                                                                      all_mask_paths_,
                                                                      test_size=0.05,
                                                                      random_state=1113)
    print(len(image_test), len(mask_test))
    print(image_test[100], mask_test[100])
    tf_record = TF_Record(image_shape=(256, 256, 3))
    tf_record.make_record(image_train, mask_train, record_save_path=root_dir, record_save_name='train_seg_256_7390')
    tf_record.make_record(image_test, mask_test, record_save_path=root_dir, record_save_name='test_seg_256_7390')

    dataset_ = get_tfrecord_data(img_shape=(256, 256, 3), re_size=256, batch_size=36, repeat=1,
                                 tf_record_path=os.path.join(root_dir, 'test_seg_256_7390'),
                                 buffer_size=300, norm=True, is_train_data=True)

    for index_, (i, j) in enumerate(dataset_):
        # print(i.shape)
        # print(np.max(j[0]))
        plt.imshow(i[0])
        plt.show()
        print(index_)
        print(j.shape)
        print(i.shape)
        break
    # print(all_image_paths_)
    # print(all_mask_paths_)
    # img = plt.imread(r'J:\DATA\OxfordCat\images\images\Abyssinian_34.jpg')
    # print(img)
