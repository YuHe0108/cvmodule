"""工具类"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2 as cv
import pathlib
import random
import os

data_dir = r'C:\Users\yuhe\Desktop\image_test'
save_dir = r'D:/Data/shenzhen/recyclable/res'

LABEL_NAMES = set()

for i, path in enumerate(pathlib.Path(data_dir).iterdir()):
    try:
        if path.suffix == '.jpg' or path.suffix == '.png':
            name = path.stem
            if path.suffix == '.jpg':
                xml_path = str(path).replace('jpg', 'xml')
            elif path.suffix == '.png':
                xml_path = str(path).replace('png', 'xml')

            if not os.path.exists(xml_path):
                continue
            image = plt.imread(str(path))
            img_h, img_w = image.shape[:2]

            tree = ET.parse(xml_path)
            root = tree.getroot()
            for member in root.findall('object'):
                obj_type = member[0].text
                LABEL_NAMES.add(obj_type)
                obj_type_idx = str(1)

                x1 = int(member[4][0].text)
                y1 = int(member[4][1].text)
                x2 = int(member[4][2].text)
                y2 = int(member[4][3].text)
                w = x2 - x1
                h = y2 - y1
    except Exception as e:
        print(e)

print(LABEL_NAMES)