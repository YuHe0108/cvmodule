import os
import pathlib

class_names = set()

for path in pathlib.Path(
        r'/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/val_data/calc_map_res/predict_txt').iterdir():

    if path.suffix != '.txt':
        continue

    result_list = []
    with open(str(path), 'r', encoding='utf-8') as file:
        for line in file.readlines():
            class_names.add(line.split(' ')[0])
            if line.split(' ')[0] != '7':
                result_list.append(line)
    # print(result_list)
    with open(str(path), "w") as f:
        f.writelines(result_list)

print(sorted(class_names))
