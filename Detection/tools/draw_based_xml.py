"""根据 xml 文件绘制目标框"""
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2
import os

data_dir = '/mnt/YuHe/data/SDYD/left/train_data_0919'
save_dir = '/mnt/YuHe/data/SDYD/left/draw_res'

# idx_to_label = {0: "full-trash-bag", 1: 'plastic-bag', 2: 'napkin',
#                 3: 'color-packing', 4: 'kraft', 5: 'bottle', 6: 'can', 7: 'other'}
# idx_to_label = {0: "glass", 1: 'other'}
idx_to_label = {0: "glass", 1: 'metal', 2: 'plastic', 3: 'paper', 4: 'full-trash-bag', 5: 'other'}

for i, path in enumerate(pathlib.Path(data_dir).iterdir()):
    try:
        if path.suffix == '.jpg' or path.suffix == '.png' or path.suffix == '.jpeg':
            name = path.stem
            if path.suffix == '.jpg':
                xml_path = str(path).replace('jpg', 'xml')
            elif path.suffix == '.png':
                xml_path = str(path).replace('png', 'xml')
            else:
                xml_path = str(path).replace('jpeg', 'xml')

            if not os.path.exists(xml_path):
                continue
            image = plt.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w = image.shape[:2]

            tree = ET.parse(xml_path)
            root = tree.getroot()
            for member in root.findall('object'):
                obj_type = member[0].text
                obj_type_idx = str(1)

                x1 = int(member[4][0].text)
                y1 = int(member[4][1].text)
                x2 = int(member[4][2].text)
                y2 = int(member[4][3].text)
                w = x2 - x1
                h = y2 - y1

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, obj_type, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            print(image.shape)
            cv2.imwrite(os.path.join(save_dir, f'{path.name}'), image)
    except Exception as e:
        print(e)
