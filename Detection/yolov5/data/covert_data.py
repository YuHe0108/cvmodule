import os
import pathlib
from sklearn.model_selection import train_test_split


def convert(data_dir, test_size=0.2, suffix=None):
    """遍历 data_dir 下所有以suffix结尾的文件"""
    if suffix is None:
        suffix = ('.jpg', '.png', '.jpeg')
    elif isinstance(suffix, str):
        suffix = [suffix]

    files = []
    for file in pathlib.Path(data_dir).iterdir():
        if file.suffix in suffix:
            files.append(str(file))

    fake_y = [i for i in range(len(files))]
    train, test, _, _ = train_test_split(files, fake_y, test_size=test_size)

    save_dir = pathlib.Path(data_dir).parent
    with open(os.path.join(save_dir, 'train.txt'), 'w') as file:
        for f in train:
            file.write(f + '\n')
    with open(os.path.join(save_dir, 'test.txt'), 'w') as file:
        for f in test:
            file.write(f + '\n')
    return


if __name__ == '__main__':
    convert(r'D:\LasoFiles\Github\DL\yolov5-master_elephant\data\images')
