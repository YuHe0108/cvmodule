import os
import cv2
import tqdm
import json
import pathlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def train_test_val_split_random(img_paths, ratio_train=0.8, ratio_test=0.1, ratio_val=0.1):
    assert int(ratio_train + ratio_test + ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths, test_size=1 - ratio_train, random_state=233)
    ratio = ratio_val / (1 - ratio_train)
    val_img, test_img = train_test_split(middle_img, test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def coco2yolo(path_dir, save_dir):
    return


def yolo2coco(root_dir, save_dir, random_split=0.1):
    """
    拆分数据集为 训练集、验证集
    """
    assert os.path.exists(root_dir)
    label_dir = os.path.join(root_dir, 'labels')  # 存放标签的位置
    image_dir = os.path.join(root_dir, 'images')  # 存放图像的位置
    with open(os.path.join(root_dir, 'classes.txt')) as f:
        classes = f.read().strip().split()  # 读取类别

    indexes = os.listdir(image_dir)
    dataset = {}
    train_dataset = val_dataset = test_dataset = None
    train_img = val_img = test_img = None
    if random_split > 0:
        # 用于保存所有数据的图片信息和标注信息
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # 建立类别标签和数字id的对应关系, 类别id从0开始。
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

        train_img, val_img, test_img = train_test_val_split_random(
            indexes, 1 - random_split * 2, random_split, random_split)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm.tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txt_file = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
        # 读取图像的宽和高
        im = plt.imread(os.path.join(root_dir, 'images/') + index)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        height, width, _ = im.shape
        if random_split > 0:  # 数据集划分
            if index in train_img:
                dataset = train_dataset
            elif index in val_img:
                dataset = val_dataset
            elif index in test_img:
                dataset = test_dataset
        # 添加图像的信息
        dataset['images'].append({
            'file_name': index, 'id': k, 'width': width, 'height': height})
        if not os.path.exists(os.path.join(label_dir, txt_file)):  # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(label_dir, txt_file), 'r') as fr:
            label_list = fr.readlines()
            for label in label_list:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_dir, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if random_split > 0:
        for phase in ['train', 'val', 'test']:
            json_name = os.path.join(root_dir, 'annotations/{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_dir, 'annotations/{}'.format(save_dir))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

    print('Convert yolo to coco ok!')
    return
