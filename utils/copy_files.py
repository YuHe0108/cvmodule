import os
import pathlib
import shutil
import argparse


def parse_opt(known=False):
    parse = argparse.ArgumentParser()
    parse.add_argument("--root", type=str, default='', help="测试")
    parse.add_argument("--save_root", type=str, default='', help="测试")
    opt = parse.parse_known_args()[0] if known else parse.parse_args()
    return opt


def run():
    opt = parse_opt(True)

    if not os.path.exists(opt.save_root):
        os.mkdir(opt.save_root)

    for path in pathlib.Path(opt.root).iterdir():
        shutil.copy(str(path), opt.save_root)
    print("file move completed!")
    return


if __name__ == '__main__':
    run()
