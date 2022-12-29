"""解析 coco 数据集"""

from collections import defaultdict
import shutil
import json
import cv2
import os

# https://blog.csdn.net/weixin_50727642/article/details/122892088

coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

name2idx = {'dog': 0, 'cat': 1, 'person': 3, 'car': 4, 'bus': 4, 'truck': 4, 'motorcycle': 5, 'umbrella': 6}


def parse_data(json_path, img_root, save_txt_root, save_img_root, class_name):
    with open(json_path, 'r', encoding='utf-8') as file:
        info = json.load(file)
    print(info.keys())

    annotations = info['annotations']
    image_info = info['images']
    print(len(annotations), len(image_info))

    # 记录每张图的标注信息
    record_info = defaultdict(list)
    for anno in annotations:
        img_id = "{:012d}".format(anno['image_id'])
        box = anno['bbox']
        cate_id = int(anno['category_id'])
        if cate_id not in class_name:
            continue
        idx = name2idx[cate_id]  #
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        # 读取图像
        img = cv2.imread(os.path.join(img_root, f'{img_id}.jpg'))
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        # cv2.imshow(f'img_{idx}', img)
        # cv2.waitKey(0)
        img_h, img_w = img.shape[:2]
        # x1, y1, w, h = x1 / img_w, y1 / img_h, w / img_w, h / img_h
        xc, yc, w, h = (x2 - x1) / 2 / img_w, (y2 - y1) / 2 / img_h, w / img_w, h / img_h
        record_info[img_id].append([idx, xc, yc, w, h])

    # 写入到txt, 并复制图像
    for img_id, info in record_info.items():
        values = []
        for v in info:
            index, xc, yc, w, h = v
            line = " ".join([str(index), str(xc), str(yc), str(w), str(h)]) + "\n"
            values.append(line)

        # 写入txt
        txt_path = os.path.join(save_txt_root, f'{img_id}.txt')
        with open(txt_path, "w") as f:
            f.writelines(values)
        # 复制图像
        shutil.copy(os.path.join(img_root, f"{img_id}.jpg"), os.path.join(save_img_root, f"{img_id}.jpg"))
    return


if __name__ == '__main__':
    parse_data(json_path=r'/mnt/YuHe/data/SDYD/cat_dog/useful/coco/annotations_trainval2017/annotations/instances_train2017.json',
               img_root=r'/mnt/YuHe/data/SDYD/cat_dog/useful/coco/train2017',
               save_txt_root=r'/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/txt',
               save_img_root=r'/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/images')
