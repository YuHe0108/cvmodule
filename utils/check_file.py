import pathlib
import os

img_root = '/mnt/YuHe/data/SDYD/left/detection/train_val_data/images/train'
txt_root = '/mnt/YuHe/data/SDYD/left/detection/train_val_data/labels/train'
for path in pathlib.Path(img_root).iterdir():
    name = path.stem
    txt_path = os.path.join(txt_root, f"{name}.txt")
    if not os.path.exists(txt_path):
        print(path)
