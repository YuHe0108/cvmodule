""""根据 txt 返回每个类别预测框的数量"""
from collections import defaultdict
import pathlib
import os


def statistical_txt(root_dir):
    """根据 txt 文件，统计每个类别预测框的数量"""
    count = defaultdict(int)
    class_names = set()
    for path in pathlib.Path(root_dir).iterdir():
        if path.suffix != '.txt':
            continue
        with open(str(path), 'r', encoding='utf-8') as file:
            for line in file.readlines():
                class_names.add(line.split(' ')[0])
                count[int(line.split(' ')[0])] += 1
    return count


if __name__ == '__main__':
    a = statistical_txt(r'C:\Users\yuhe\Desktop\valid_data\predict\ori_pred\0902-1\1')
    print(a)
