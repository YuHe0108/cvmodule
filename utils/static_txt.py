import os
import pathlib
from collections import defaultdict

class_names = set()
count = defaultdict(int)
# names = [
#     'cat',
#     'person',
#     'broom',
#     'dustpan',
#     'trash',
#     'clip',
#     'storage-battery',
#     'car',
#     'bucket',
#     'bike',
#     'ladder',
#     'mop',
#     'bench',
#     'dog',
#     'stool',
#     'umbrella',
#     'tricycle',
#     'spade',
#     'chair',
#     'trolley',
#     'wash-basin',
#     'strollers',
#     'bird',
# ]
names = ['non', 'left', 'ground', 'waste']
for path in pathlib.Path(
        r'/mnt/YuHe/data/SDYD/left/detection/train_val_data/labels/train').iterdir():
    if path.suffix != '.txt':
        continue
    with open(str(path), 'r', encoding='utf-8') as file:
        for line in file.readlines():
            class_names.add(line.split(' ')[0])
            idx = int(line.split(' ')[0])   
            count[names[idx]] += 1

print(sorted(class_names))
print(count)

for i, name in enumerate(names):
    print(count[name])


# keys = ['person', 'car', 'trash', 'storage-battery', 'bucket', 'dog', 'broom', 'stool', 'dustpan', 'tricycle', 'cat', 'clip',
#         'mop', 'bench', 'trolley', 'spade', 'wash-basin', 'bike', 'umbrella', 'chair', 'strollers', 'ladder']
# for k in keys:
#     print(count[k])
