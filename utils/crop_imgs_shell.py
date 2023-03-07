"""根据 xml 文件：裁剪图像"""
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import threading
import datetime
import argparse
import pathlib
import random
import glob
import json
import uuid
import tqdm
import time
import cv2
import os

lock = threading.Lock()
RATE = 0.1
SAVE_SINGLE_FOLDER = False


def parse_opt(known=False):
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_root", type=str, default='')
    parse.add_argument("--save_root", type=str, default='')
    parse.add_argument("--label_path", type=str, default='')
    parse.add_argument("--valid_rate", type=float, default=0.1)
    parse.add_argument("--num_workers", type=int, default=12)
    parse.add_argument("--shift_rate", type=float, default=0.2)
    parse.add_argument("--resize_shape", type=list, default=[224, 224])
    parse.add_argument("--save_single_folder", default=False, action="store_true")  # 默认false
    parse.add_argument("--resize_data",  default=False, action="store_true")  # 默认true
    opt = parse.parse_known_args()[0] if known else parse.parse_args()
    return opt


def label_statistic(dataset_dir, label2int):
    label_dict = defaultdict(int)
    for xml_file in glob.glob(os.path.join(dataset_dir, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)

        if len(root.findall('object')) == 0:
            continue
        for member in root.findall('object'):
            x_min = int(member[4][0].text)
            y_min = int(member[4][1].text)
            x_max = int(member[4][2].text)
            y_max = int(member[4][3].text)
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > width:
                x_max = width
            if y_max > height:
                y_max = height
            label = member[0].text
            if x_min >= x_max or y_min >= y_max or label not in label2int:
                print("error data:{},file name:{} label:{}".format([x_min, y_min, x_max, y_max], xml_file, label))
                continue
            label_dict[label2int[label]] += 1
    return label_dict


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """缩放图像尺寸至： new_shape"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def join_together(img, shape):
    # 通过拼接的方式，将图像
    # ori = part.copy()
    # ori_h, ori_w = ori.shape[:2]
    # h, w = part.shape[:2]
    # size = 2
    # while h < shape[0] or w < shape[1]:
    #     part = np.zeros(shape=(ori_h * size, ori_w * size, 3), dtype=np.uint8)
    #     for i in range(size):
    #         for j in range(size):
    #             part[i * ori_h: (i + 1) * ori_h, j * ori_w:(j + 1) * ori_w] = ori
    #     h, w = part.shape[:2]
    #     size += 1
    out = img
    h, w = out.shape[:2]
    ori_h, ori_w = img.shape[:2]
    while h < shape[0] or w < shape[1]:
        out = cv2.copyMakeBorder(out, ori_h, 0, ori_w, 0, cv2.BORDER_WRAP)
        h, w = out.shape[:2]
    return out[0:224, 0:224]


def join_together_experimental(img, shape):
    hwc = img.shape
    h = hwc[0]
    w = hwc[1]
    const_w = 224
    const_h = 224
    if w < const_w or h < const_h:
        if w > const_w:
            const_w = w
            const_h = w
        if h > const_h:
            const_w = h
            const_h = h
        sub_w = const_w - w
        sub_h = const_h - h
        # 水平翻转图像
        img_v = cv2.flip(img, 1)
        index = 0
        img0 = img.copy()
        while sub_w > 0:

            img2 = img
            if index % 2 == 0:
                img2 = img_v
            index += 1

            if sub_w < w:
                img3 = img2[0:h, 0:sub_w]
                img0 = np.hstack([img0, img3])
                break
            else:
                img0 = np.hstack([img0, img2])
            sub_w = sub_w - w

        index = 0
        # 垂直翻转
        img_h = cv2.flip(img0, 0)
        img4 = img0.copy()
        w4 = img0.shape[1]
        while sub_h > 0:
            img2 = img4
            if index % 2 == 0:
                img2 = img_h
            index += 1
            if sub_h < h:
                img3 = img2[0:sub_h, 0:w4]
                img0 = np.vstack([img0, img3])
                break
            else:
                img0 = np.vstack([img0, img2])
            sub_h = sub_h - h
    else:
        img0 = letterbox(img, shape)
        img0 = join_together(img0, shape)
    return img0


def crop_img(img_paths,
             xml_paths,
             name2int,
             augment_threshold_nums,
             type_cnt_dict,
             aug_times,
             resize_shape,
             train_save_dir,
             valid_save_dir):
    global lock, RATE
    rate = RATE  # 图像增强时，前后移动的比例

    cur_date = f'{datetime.datetime.today().date()}'
    for i, img_path in enumerate(img_paths):
        xml_path = xml_paths[i]
        image = plt.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        # 解析 xml 文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            with lock:  # 加锁
                obj_type = member[0].text.lower()
                # try:
                if obj_type not in name2int:
                    print("NOT FIND THIS LABEL: ", obj_type)
                    continue

                # 裁减
                obj_type_idx = str(name2int[obj_type])
                x1 = int(member[4][0].text)
                y1 = int(member[4][1].text)
                x2 = int(member[4][2].text)
                y2 = int(member[4][3].text)
                w = x2 - x1
                h = y2 - y1

                # 判断是训练还是验证集
                batch = aug_times[int(obj_type_idx)]
                if type_cnt_dict[int(obj_type_idx)] < augment_threshold_nums[int(obj_type_idx)]:
                    save_dir = valid_save_dir
                else:
                    save_dir = train_save_dir

                # 进行数据增强
                for _ in range(int(batch)):
                    h_range = max(1, int(h * rate))
                    w_range = max(1, int(w * rate))
                    range_val = [random.randrange(-h_range, h_range, 1),
                                 random.randrange(-w_range, w_range, 1),
                                 random.randrange(-h_range, h_range, 1),
                                 random.randrange(-w_range, w_range, 1)]
                    # 截图
                    part = image[
                           max(0, int(y1 + range_val[0])):min(int(y2 + range_val[2]), img_h),
                           max(0, int(x1 + range_val[1])):min(int(x2 + range_val[3]), img_w)]
                    cur_cnt = type_cnt_dict[int(obj_type_idx)]

                    # 每一个类别新建一个文件夹保存
                    if not os.path.exists(os.path.join(save_dir, obj_type_idx)):
                        os.makedirs(os.path.join(save_dir, obj_type_idx))

                    # 截取图像的大小
                    p_h, p_w = part.shape[:2]
                    if p_h < 20 or p_w < 20:  # 较小的直接忽略
                        continue
                    # 将截取的图像进行 resize、保存
                    part = letterbox(part, resize_shape, (0, 0, 0))
                    # part = join_together_experimental(part, resize_shape)
                    plt.imsave(os.path.join(os.path.join(save_dir, obj_type_idx), f'{str(uuid.uuid4())}_{cur_date}.jpg'), part)
                    type_cnt_dict[int(obj_type_idx)] += 1
                # except Exception as e:
                #     print("crop_img run exception: ", e, obj_type)
    return


def run_crop(data_root, save_root, name2int_json_path, valid_rate, resize_shape, num_workers=10):
    """
    data_root:  图像和对应的 xml 文件保存在一起
    valid_num:  每个类别需要多少张验证集
    aug_times:  训练集每个类别需要增强多少倍
    """
    global SAVE_SINGLE_FOLDER
    # 图像的保存位置
    if not SAVE_SINGLE_FOLDER:
        train_save_dir = os.path.join(save_root, 'train')
        valid_save_dir = os.path.join(save_root, 'valid')
    else:
        train_save_dir = valid_save_dir = save_root

    if os.path.exists(train_save_dir):
        shutil.rmtree(train_save_dir)
    os.makedirs(train_save_dir)
    if os.path.exists(valid_save_dir):
        shutil.rmtree(valid_save_dir)
    os.makedirs(valid_save_dir)

    with open(name2int_json_path) as file:
        name2int = json.load(file)

    # 标签统计
    aug_times = {}  # 每个类别增强几张图
    label_count = label_statistic(data_root, name2int)
    max_val = max(label_count.values())
    for k, v in label_count.items():
        if v == max_val:
            aug_times[k] = 2
        else:
            aug_times[k] = 2 * max_val // v + 1

    # 数据增强的参数
    type_cnt_dict = {i: 0 for i in range(len(label_count))}
    valid_nums = {i: int(label_count[i] * valid_rate) for i in range(len(label_count))}
    augment_threshold_nums = {i: int(aug_times[i]) * valid_nums[i] for i in range(len(label_count))}
    print(f"标签数量统计: {label_count}, 标签增强倍数为: {aug_times}, 验证集标签数量: {augment_threshold_nums}")

    # 从目录下逐个读取文件夹
    have_sub_dir = False
    for path in pathlib.Path(data_root).iterdir():
        if path.is_dir():
            have_sub_dir = True

    data_paths = []
    if not have_sub_dir:
        data_paths = [data_root]
    else:
        for data_path in pathlib.Path(data_root).iterdir():
            if data_path.is_dir():
                data_paths.append(str(data_path))

    # 逐个文件夹读取图像文件
    miss_xml_cnt = 0
    img_paths = []
    xml_paths = []
    img_suffixes = {".jpg", ".JPG", ".PNG", ".png"}
    for data_path in tqdm.tqdm(data_paths, total=len(data_paths)):
        print(data_path)
        total = 0
        for path in tqdm.tqdm(pathlib.Path(data_path).iterdir()):
            try:
                if path.suffix in img_suffixes:
                    xml_path = str(path).replace(path.suffix, '.xml')
                    if not os.path.exists(xml_path):
                        miss_xml_cnt += 1
                        print(f"当前图 {str(path)} 缺少 xml 文件.")
                        continue
                    img_paths.append(str(path))
                    xml_paths.append(str(xml_path))
                    total += 1
                elif path.suffix == '.xml':
                    continue
                else:
                    print(f"当前图像后缀未设置：{path.suffix}")
                    return
            except Exception as e:
                print("exception", str(path), e)

    # shuffle
    union_data = list(zip(img_paths, xml_paths))
    random.shuffle(union_data)
    img_paths, xml_paths = list(zip(*union_data))

    batch = len(img_paths) // num_workers
    tasks = []
    for cur_worker in range(num_workers):
        cur_img_paths = img_paths[cur_worker * batch:(cur_worker + 1) * batch]
        cur_xml_paths = xml_paths[cur_worker * batch:(cur_worker + 1) * batch]
        if cur_worker + 1 == num_workers:
            cur_img_paths = img_paths[cur_worker * batch:]
            cur_xml_paths = xml_paths[cur_worker * batch:]

        task = threading.Thread(
            target=crop_img, name=str(cur_worker),
            args=(cur_img_paths, cur_xml_paths, name2int, augment_threshold_nums,
                  type_cnt_dict, aug_times, resize_shape, train_save_dir, valid_save_dir))
        task.start()
        tasks.append(task)

    for task in tasks:
        task.join()
    print('miss xml nums: ', miss_xml_cnt)
    print("总计：", type_cnt_dict)
    return


def run():
    global RATE, SAVE_SINGLE_FOLDER
    opt = parse_opt(True)
    RATE = opt.shift_rate
    SAVE_SINGLE_FOLDER = opt.save_single_folder
    print(opt.resize_data)
    t = time.time()
    run_crop(data_root=opt.data_root,
             save_root=opt.save_root,
             name2int_json_path=opt.label_path,
             valid_rate=opt.valid_rate,
             resize_shape=opt.resize_shape,
             num_workers=opt.num_workers)
    print("completed, cost time: ", time.time() - t, 'time cost')
    return


if __name__ == '__main__':
    run()
