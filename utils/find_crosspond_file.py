import argparse
import pathlib
import shutil
import cv2
import os


def parse_opt(known=False):
    parse = argparse.ArgumentParser()
    parse.add_argument("--img_root", type=str, default='/mnt/YuHe/data/SDYD/left/classify/history/20230224/val', help="测试")
    parse.add_argument("--xml_dir", type=str, default='/mnt/YuHe/data/SDYD/left/detection/history/20230224/raw', help="测试")
    opt = parse.parse_known_args()[0] if known else parse.parse_args()
    return opt


def find_file_1(ori_dir, img_dir, txt_dir, save_dir):
    for path in pathlib.Path(ori_dir).iterdir():
        name = path.stem
        img_path = os.path.join(img_dir, f'{name}.jpg')
        txt_path = os.path.join(txt_dir, f'{name}.txt')
        print(img_path)
        print(os.path.exists(img_path))
        shutil.copy(img_path, save_dir)
        shutil.copy(txt_path, save_dir)
    return


def find_file(txt_dir, img_dir, xml_dir, save_dir):
    cnt = 0
    for path in pathlib.Path(img_dir).iterdir():
        if path.suffix == '.xml':
            continue
        txt_path = os.path.join(txt_dir, path.stem + '.txt')
        xml_path = os.path.join(xml_dir, path.stem + '.xml')
        if os.path.exists(txt_path):  # 当前文件在验证集里面
            cnt += 1
            shutil.copy(str(path), save_dir)
            shutil.copy(xml_path, save_dir)
    print(cnt)


def find_xml_file(img_dir, ori_xml_dir):
    img_suffix = ['.jpg', '.png']
    for path in pathlib.Path(img_dir).iterdir():
        if path.suffix in img_suffix:
            name = path.stem
            xml_path = os.path.join(ori_xml_dir, f'{name}.xml')
            if os.path.exists(xml_path):
                shutil.copy(xml_path, img_dir)
            else:
                print(f'find file error: {str(path)}')
        else:
            print("图像文件格式有误: ", path)
            continue
    return


# txt_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/waste_0901/labels/val'
# img_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/previous_data/raw'
# xml_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/previous_data/raw'
# save_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/val_data/calc_map_huzhou_val_data/huzhou_val_data'
# find_file(txt_dir_, img_dir_, xml_dir_, save_dir_)
# 2934
# find_xml_file(r'/mnt/YuHe/data/val_data/tools/val',
#               r'/mnt/YuHe/data/SDYD/tools/train_val_data/raw')

# find_file_1(r'/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/draw',
#             '/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/images',
#             '/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/txt',
#             '/mnt/YuHe/data/SDYD/cat_dog/useful/coco/raw/coco_tools')
# find_xml_file('/mnt/YuHe/data/val_data/tools/v1-valdata',
#               '/mnt/YuHe/data/SDYD/tools/history/coco/raw')


def move_file(dir1, dir2):
    img_suffix = ['.jpg', '.png']
    for path in pathlib.Path(dir1).iterdir():
        name = path.stem
        if path.suffix in img_suffix:
            xml_name = f'{name}.xml'
            if not os.path.exists(os.path.join(dir1, xml_name)):
                shutil.move(str(path), dir2)
    return


def run():
    opt = parse_opt()
    find_xml_file(opt.img_root, opt.xml_dir)
    return


if __name__ == '__main__':
    run()
    # find_xml_file("/mnt/YuHe/data/SDYD/left/classify/history/20230216/val",
    #               "/mnt/YuHe/data/SDYD/left/detection/history/20230216/raw")