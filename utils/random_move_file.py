import os
import pathlib
import shutil
import random

file_name = 7
img_dir = f'/mnt/YuHe/data/SDYD/left/multi_classes/train_val_data/train/{file_name}'
xml_dir = '/mnt/YuHe/data/shenzhen/recyclable/train_val_data/raw'
# xml_dir = '/mnt/YuHe/data/huzhou/history/huzhou_previous_data/raw'
# img_dir = '/mnt/YuHe/data/val_data/huzhou/valid/3'
# xml_dir = '/mnt/YuHe/data/val_data/huzhou/valid/3'
move_xml_dir = f'/mnt/YuHe/data/SDYD/left/multi_classes/train_val_data/valid/{file_name}'
move_img_dir = move_xml_dir

img_suffixes = ['.jpg', '.png']

if __name__ == '__main__':
    img_names = []
    for path in pathlib.Path(img_dir).iterdir():
        if path.suffix in img_suffixes:
            img_names.append(path.stem)

    nums = 250
    cnt = 0
    while cnt < nums:
        choices = random.sample(img_names, nums)
        miss_count = 0
        for i, path in enumerate(pathlib.Path(img_dir).iterdir()):
            file_name = path.stem
            if file_name not in choices or path.suffix not in img_suffixes:
                continue
            xml_name = file_name + '.xml'
            # xml_path = os.path.join(xml_dir, xml_name)
            # if not os.path.exists(xml_path):
            #     miss_count += 1
            #     continue

            if nums == cnt:
                break
            # if os.path.exists(os.path.join(move_xml_dir, xml_name)):
            #     continue
            shutil.move(str(path), move_img_dir)
            # shutil.copy(xml_path, move_xml_dir)
            cnt += 1
        print(cnt)
