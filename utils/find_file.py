import pathlib
import shutil
import os

file_dir = '/mnt/YuHe/data/val_data/tools/v6-valdata-add'
xml_path = "/mnt/YuHe/data/SDYD/tools/history/20230112/raw"
for path in pathlib.Path(file_dir).iterdir():
    name = path.stem
    file_path = os.path.join(xml_path, name + ".xml")
    if os.path.exists(file_path):
        shutil.copy(file_path, file_dir)
        print(name)
