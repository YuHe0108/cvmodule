"""根据 xml 文件：裁剪图像"""
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pathlib
import random
import json
import tqdm
import cv2
import os


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


def run_crop(data_root, save_root, name2int_json_path, valid_num, aug_times, resize_shape, base_num=1000000):
    """
    data_root:  图像和对应的 xml 文件保存在一起
    valid_num:  每个类别需要多少张验证集
    aug_times:  训练集每个类别需要增强多少倍
    """
    rate = 0.2  # 图像增强时，前后移动的比例

    with open(name2int_json_path) as file:
        name2int = json.load(file)

    # 图像的保存位置
    train_save_dir = os.path.join(save_root, 'train')
    valid_save_dir = os.path.join(save_root, 'valid')
    if not os.path.exists(train_save_dir):
        os.mkdir(train_save_dir)
    if not os.path.exists(valid_save_dir):
        os.mkdir(valid_save_dir)

    # 数据增强的参数
    random_augment = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}  # 随机设置增强的概率
    type_cnt_dict = {i: base_num * (i + 1) for i in range(10)}
    augment_threshold_nums = {i: base_num * (i + 1) + valid_num.get(i, 0) for i in range(10)}
    batch_nums = {i: aug_times.get(i, 1) for i in range(10)}  # 每个类别增强几张图

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
    for data_path in tqdm.tqdm(data_paths, total=len(data_paths)):
        for path in tqdm.tqdm(pathlib.Path(data_path).iterdir()):
            try:
                if path.suffix == '.jpg' or path.suffix == '.png':
                    if path.suffix == '.jpg':
                        xml_path = str(path).replace('jpg', 'xml')
                    elif path.suffix == '.png':
                        xml_path = str(path).replace('png', 'xml')
                    else:
                        print(f"当前图像后缀未设置：{path.suffix}")
                        return

                    if not os.path.exists(xml_path):
                        print("当前图没有 xml 文件", xml_path)
                        continue

                    image = plt.imread(str(path))
                    img_h, img_w = image.shape[:2]

                    # 解析 xml 文件
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for member in root.findall('object'):
                        obj_type = member[0].text.lower()
                        try:
                            obj_type_idx = str(name2int[obj_type])
                            x1 = int(member[4][0].text)
                            y1 = int(member[4][1].text)
                            x2 = int(member[4][2].text)
                            y2 = int(member[4][3].text)
                            w = x2 - x1
                            h = y2 - y1

                            # 根据类型
                            if type_cnt_dict[int(obj_type_idx)] < augment_threshold_nums[int(obj_type_idx)]:  # 验证集的保存
                                batch = 1
                                random_val = 0
                                save_dir = valid_save_dir
                            else:  # 训练集图像保存
                                save_dir = train_save_dir
                                batch = batch_nums[int(obj_type_idx)]
                                random_val = random_augment[int(obj_type_idx)]

                            # 进行数据增强
                            for i in range(batch):
                                h_range = max(1, int(h * rate))
                                w_range = max(1, int(w * rate))
                                range_val = [random.randrange(-h_range, h_range, 1),
                                             random.randrange(-w_range, w_range, 1),
                                             random.randrange(-h_range, h_range, 1),
                                             random.randrange(-w_range, w_range, 1)]
                                # range_val = [random.randrange(-h_range, 0, 1),
                                #              random.randrange(-w_range, 0, 1),
                                #              random.randrange(0, h_range, 1),
                                #              random.randrange(0, w_range, 1)]
                                # 截图
                                part = image[
                                       max(0, int(y1 + range_val[0])):min(int(y2 + range_val[2]), img_h),
                                       max(0, int(x1 + range_val[1])):min(int(x2 + range_val[3]), img_w)]
                                cur_cnt = type_cnt_dict[int(obj_type_idx)]

                                # 每一个类别新建一个文件夹保存
                                if not os.path.exists(os.path.join(save_dir, obj_type_idx)):
                                    os.mkdir(os.path.join(save_dir, obj_type_idx))

                                # 截取图像的大小
                                p_h, p_w = part.shape[:2]
                                if p_h < 20 or p_w < 20:  # 较小的直接忽略
                                    continue
                                if random.random() <= random_val:  # TODO: 取消图像通过概率进行增强
                                    continue

                                # 将截取的图像进行 resize、保存
                                part = letterbox(part, resize_shape, (0, 0, 0))
                                plt.imsave(os.path.join(os.path.join(save_dir, obj_type_idx), f'{cur_cnt}.jpg'), part)
                                type_cnt_dict[int(obj_type_idx)] += 1

                        except Exception as e:
                            print("exception..", e, obj_type)

            except Exception as e:
                print("exception", str(path), e)
    return


if __name__ == '__main__':
    run_crop(data_root=r'C:\Users\yuhe\Desktop\valid_data\1',
             save_root=r'C:\Users\yuhe\Desktop\save',
             name2int_json_path='huzhou.json',
             valid_num={0: 800, 1: 800, 2: 800},
             aug_times={0: 3, 1: 3, 2: 3},
             resize_shape=[224, 224],
             base_num=1000000)
