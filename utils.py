import math
import os
import pathlib
import shutil

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def move_all_files(root_dir, save_dir):
    """
    将一个文件夹下的所有图像全部都复制到另一个文件夹下。
    """
    files = list_all_file(root_dir)
    for i, file in tqdm.tqdm(enumerate(files)):
        shutil.copyfile(os.path.join(root_dir, file),
                        os.path.join(save_dir, f'{i}.jpg'))
    return


def copy_dir_by_name(root_dir, save_dir, dir_names):
    """
    root_dir: 原始文件夹目录
    save_dir: 带保存的目录
    dir_names: 从root_dir下复制哪些文件夹
    dir_names = "7 11 45 43 22 42 6 4 48 41 21"
    """
    dir_names = dir_names.split(' ')
    for dir_stem in tqdm.tqdm(dir_names):
        shutil.copytree(os.path.join(root_dir, dir_stem), os.path.join(save_dir, dir_stem))
    return


def copy_file_by_name(name_root_path, source_path, save_path):
    """从一个文件夹下通过文件名复制文件
        name_root_path: 根据此文件夹下文件的名称复制
        source_path: 从哪个文件夹下复制
        save_path: 将复制的文件保存在哪里
    """
    path, stem = get_path_and_name(name_root_path)
    for s in tqdm.tqdm(stem):
        save_path = os.path.join(save_path, s)
        if not os.path.exists(save_path):
            s_path = os.path.join(source_path, s)
            shutil.copyfile(s_path, save_path)
    return


def norm_file_name(file_dir):
    """更改文件夹的名称，以7z结尾"""
    file_list = list_file(file_dir)
    file_parent = pathlib.Path(file_list[0]).parent  # 文件的主目录
    for file_path in tqdm.tqdm(file_list):
        file_name_list = pathlib.Path(file_path).name.split('.')
        file_suffix = file_name_list[1][1:]
        if file_suffix != 'ip':
            new_file_name = '{} {}.zip'.format(file_name_list[0], file_suffix)
        else:
            new_file_name = '{} {}.zip'.format(file_name_list[0], 0)
        shutil.move(str(file_path), os.path.join(file_parent, new_file_name))
    return


def reverse_file_name(file_dir):
    """更改文件夹的名称，以7z结尾"""
    file_list = list_file(file_dir)
    file_parent = pathlib.Path(file_list[0]).parent  # 文件的主目录
    for file_path in tqdm.tqdm(file_list):
        file_name = pathlib.Path(file_path).stem.split(' ')
        if file_name[-1] == '0':
            new_file_name = '{}.zip'.format(file_name[0])
        else:
            new_file_name = '{}.z{}'.format(file_name[0], file_name[1])
        shutil.move(str(file_path), os.path.join(file_parent, new_file_name))
    return


def return_inputs(inputs):
    """根据输入类型返回输出值，用于图像path输入"""
    if type(inputs) is str:
        if os.path.isfile(inputs):
            all_image_paths = [inputs]
        elif os.path.isdir(inputs):
            all_image_paths = list_file(inputs)
    elif type(inputs) is list:
        all_image_paths = inputs
    return all_image_paths


def image_rename_and_move(input_dir, save_dir, start):
    """将图像转换为灰度图并保存"""
    check_file(save_dir)
    input_file_list = return_inputs(input_dir)
    for i, file in tqdm.tqdm(enumerate(input_file_list), total=len(input_file_list)):
        shutil.copy(file, os.path.join(save_dir, '{}.png'.format(i + start)))
    return


def to_gray_image(input_dir, save_dir, start):
    """将图像转换为灰度图并保存"""
    check_file(save_dir)
    input_file_list = return_inputs(input_dir)
    for i, file in tqdm.tqdm(enumerate(input_file_list), total=len(input_file_list)):
        image = cv.imread(file, 0)
        cv.imwrite(os.path.join(save_dir, '{}.png'.format(i + start)), image)
    return


def to_binary_image(input_dir, save_dir, threshold):
    """将图像转换为二值图并保存"""
    check_file(save_dir)
    input_file_list = return_inputs(input_dir)
    for i, file in tqdm.tqdm(enumerate(input_file_list), total=len(input_file_list)):
        file_stem = pathlib.Path(file).stem
        image = cv.imread(file, 0)
        _, result = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        cv.imwrite(
            os.path.join(save_dir, '{}.png'.format(file_stem)), result)
    return


def is_same_two_list(list_1, list_2):
    """判断两个列表是否相等，如果相同则返回True，否则返回list_1存在，list_2不存在的元素"""
    if list_1 == list_2:
        return True
    else:
        in_list_1_not_in_list_2 = [i for i in list_1 if i not in list_2]
        return in_list_1_not_in_list_2


def is_same_two_path_list(path_dir_1, path_dir_2):
    """查看两个dir下，文件是否相同，并将在path_list_1，不在path_list_2中的路径元素返回，
    两个dir下，文件名需要相同
    """
    file_paths_1, file_stems_1 = get_path_and_stem(path_dir_1)
    file_paths_2, file_stems_2 = get_path_and_stem(path_dir_2)
    in_dir_1_not_in_dir_2 = []
    for index, file_stem_1 in tqdm.tqdm(enumerate(file_stems_1)):
        if file_stem_1 not in file_stems_2:
            in_dir_1_not_in_dir_2.append(file_paths_1[index])
    if len(in_dir_1_not_in_dir_2) == 0:
        print('{}_{}元素完全相同'.format(path_dir_1, path_dir_2))
    return in_dir_1_not_in_dir_2


def get_path_and_stem(path_dir):
    """返回path_dir下所有的文件路径以及name, 没有文件后缀"""
    paths = []
    stems = []
    for path in pathlib.Path(path_dir).iterdir():
        paths.append(str(path))
        stems.append(path.stem)
    return sorted(paths), sorted(stems)


def get_path_and_name(path_dir):
    """返回path_dir下所有的文件路径以及name, 带有后缀"""
    paths = []
    names = []
    for path in pathlib.Path(path_dir).iterdir():
        paths.append(str(path))
        names.append(path.name)
    return sorted(paths), sorted(names)


def list_all_file(dir_path):
    """返回dir_path下所有的文件的相对路径"""
    paths = []
    for path in pathlib.Path(dir_path).iterdir():
        # 判断是文件还是目录
        if os.path.isfile(str(path)):
            paths.append(str(path))
        else:
            paths += list_all_file(str(path))
    return paths


def list_file(dir_path):
    """返回当前目录下的文件或文件夹"""
    paths = []
    for path in pathlib.Path(dir_path).iterdir():
        paths.append(str(path))
    return paths


def is_parent_dir(input_dir):
    """判断输入的文件目录下存放的是目录还是文件"""
    paths, stems = get_path_and_stem(input_dir)
    if os.path.isdir(paths[0]):
        return True
    else:
        return False


def check_file(paths):
    """检查文件是否存在"""
    if type(paths) is not list:
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return


def delete_file(dir_path):
    """删除目录下所有的文件"""
    for path in pathlib.Path(dir_path).iterdir():
        os.remove(path)
    return


def de_norm(norm_img, denorm_rang='-11'):
    """反归一化: 将图像从[-1, 1] --> [0, 255]"""
    if denorm_rang == '-11':
        img = (norm_img + 1) * 127.5
    else:
        img = norm_img * 255.0
    img = np.array(img)
    img = img.astype(np.uint8)
    return img


def make_gird(inputs, padding=2, n_rows=6, denorm=False, denorm_rang='-11'):
    """将一个batch的图像合并为一整个大图"""
    if denorm:
        de_norm_img = de_norm(inputs, denorm_rang)
    else:
        de_norm_img = inputs

    # 判断是否是灰度图
    shape_len = len(de_norm_img.shape)
    gray_image = True if (shape_len == 4 & de_norm_img.shape[-1] != 1) or shape_len == 3 else False
    if gray_image:
        n_maps, x_maps, y_maps = de_norm_img.shape  # [b, h, w, c]
        if shape_len == 4:
            de_norm_img = np.squeeze(de_norm_img, axis=-1)
    else:
        n_maps, x_maps, y_maps, c_maps = de_norm_img.shape  # [b, h, w]

    x_maps = min(n_rows, x_maps)  # 取最小值
    y_maps = int(math.ceil(float(n_maps) / x_maps))

    # 图像经过 padding 之后的高度和宽度
    height, width = int(de_norm_img.shape[1] + padding), int(de_norm_img.shape[2] + padding)

    # 初始化一整张大图
    if gray_image:
        grid = np.zeros(shape=(height * y_maps + padding, width * x_maps + padding))
    else:
        grid = np.zeros(shape=(height * y_maps + padding, width * x_maps + padding, c_maps), dtype=np.uint8)
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            if gray_image:
                grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = de_norm_img[k]
            else:
                grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width, :] = de_norm_img[k]
            k = k + 1
    return grid


def save_samples(gen_sample, save_path, epoch, step=0, i=0, color='rgb', denorm=True, denorm_rang='-11'):
    """保存[-1, 1]之间的图像"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if denorm:
        gen_sample = de_norm(gen_sample, denorm_rang=denorm_rang)
    img_name = 'epoch-{:05d}-step-{:05d}-num-{:03d}.png'.format(epoch, step, i)
    if color == 'rgb':
        plt.imsave(os.path.join(save_path, img_name), gen_sample)
    else:
        cv.imwrite(os.path.join(save_path, img_name), gen_sample)
    return


def rename_files(root_dir, save_root_dir):
    """重命名文件夹下各个图像的名称为数字"""
    files = list_file(root_dir)
    for file in files:
        file_name = pathlib.Path(file).name
        save_dir = os.path.join(save_root_dir, file_name)
        check_file(save_dir)
        imgs_path = list_file(file)
        for index, img_path in enumerate(imgs_path):
            shutil.copy(img_path, os.path.join(save_dir, '{}.jpg'.format(index)))
    return


def reshape_imgs(root_dir, save_root_dir, reshape_size=(800, 800)):
    """reshape各个文件夹下图像的尺寸"""
    files = list_file(root_dir)
    for file in files:
        file_name = pathlib.Path(file).name
        save_dir = os.path.join(save_root_dir, file_name)
        check_file(save_dir)
        imgs_path = list_file(file)
        for index, img_path in enumerate(imgs_path):
            img = cv.imread(img_path)
            img = cv.resize(img, reshape_size)
            cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(index)), img)
    return


if __name__ == '__main__':
    # import test_data
    #
    # cifar_data = test_data.get_data(batch_size=36, data_name='cifar10')
    # for imgs, labels in cifar_data:
    #     de_norm_img_ = make_gird(imgs, denorm=True, padding=2)
    #     print(np.max(de_norm_img_))
    #     print(np.min(de_norm_img_))
    #     print(de_norm_img_.dtype)
    #     plt.imshow(de_norm_img_)
    #     plt.axis('off')
    #     plt.show()
    #     break
    # a, b = get_path_name(r'D:\Users\YingYing\Desktop\make_mask\mask')
    # print(a)
    # print(b)
    # return_result_ = is_same_two_path_list(r'E:\tookit_backup\毕业论文\程序\Segment\data\no_skull_image',
    #                                        r'E:\tookit_backup\毕业论文\程序\Segment\fcm')
    # print(return_result_)
    # print(len(return_result_))
    # norm_file_name(r'J:\DATA\ObjDet\COCO\COCO_2017train_zip')
    reverse_file_name(r'J:\DATA\ObjDet\COCO\COCO_2017train_zip')
    pass
