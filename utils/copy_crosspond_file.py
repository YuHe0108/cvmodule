import pathlib
import shutil
import os


def copy_file(ori_dir, cp_dir, cv_dir):
    """将 ori_dir 下的文件，根据文件名，从 cp_dir 中寻找，并移动至 cv_dir"""
    cnt = 0 
    for path in pathlib.Path(ori_dir).iterdir():
        name = path.name
        cp_path = os.path.join(cp_dir, name)
        if not os.path.exists(cp_path):
            cnt += 1
            print(name)
            continue
        shutil.move(cp_path, cv_dir)
    print(cnt)
    return


if __name__ == '__main__':
    copy_file('/mnt/YuHe/data/shushan/train_val_data/labels/train',
              '/mnt/YuHe/data/shushan/history/platform/raw',
              '/mnt/YuHe/data/shushan/history/platform/labels/train')
