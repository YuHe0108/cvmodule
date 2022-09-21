import os
import pathlib
from collections import defaultdict

class_names = set()
count = defaultdict(int)

for path in pathlib.Path(
        r'/mnt/YuHe/data/shenzhen/recyclable_0913/labels/val').iterdir():
    if path.suffix != '.txt':
        continue
    with open(str(path), 'r', encoding='utf-8') as file:
        for line in file.readlines():
            class_names.add(line.split(' ')[0])
            count[line.split(' ')[0]] += 1
print(sorted(class_names))
print(count)
