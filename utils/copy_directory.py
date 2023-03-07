import argparse
import pathlib
import shutil
import os


def parse_opt(known=False):
    parse = argparse.ArgumentParser()
    parse.add_argument("--ori_root", type=str, default='/mnt/YuHe/data/SDYD/tools/train_val_data', help="原文件夹")
    parse.add_argument("--save_root", type=str, default='/mnt/YuHe/data/SDYD/tools/train_val_data-1', help="复制文件夹")
    opt = parse.parse_known_args()[0] if known else parse.parse_args()
    return opt


def copy_directory(ori_root, save_root):
    # 将文件夹写的所有文件，拷贝到另一个文件夹中
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    shutil.copytree(ori_root, save_root)
    return


def run():
    opt = parse_opt()
    copy_directory(opt.ori_root, opt.save_root)
    return


if __name__ == '__main__':
    run()
