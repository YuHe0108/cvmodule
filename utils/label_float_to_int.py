import pathlib
import cv2
import os


def float_to_int(txt_dir, img_dir, save_dir):
    for path in pathlib.Path(img_dir).iterdir():
        txt_path = os.path.join(txt_dir, path.stem + '.txt')
        img = cv2.imread(str(path))
        img_h, img_w = img.shape[:2]

        result_txt = []
        with open(str(txt_path), 'r', encoding='utf-8') as file:
            for line in file.readlines():
                val = []
                res = line.strip().split(' ')
                if res[0] == '7':
                    continue
                val.append(res[0])
                x1, y1, w, h = res[1:]
                x1 = int(float(x1) * img_w)
                y1 = int(float(y1) * img_h)
                x2 = int(float(w) * img_w)
                y2 = int(float(h) * img_h)
                val.append(str(x1))
                val.append(str(y1))
                val.append(str(x2))
                val.append(str(y2))
                val = ' '.join(val)
                result_txt.append(val + '\n')

        with open(os.path.join(save_dir, path.stem + '.txt'), "w") as f:
            f.writelines(result_txt)
    print('转换完成！')
    return


if __name__ == '__main__':
    txt_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/waste_0901/labels/val'
    img_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/waste_0901/images/val'
    root_dir_ = ''
    save_dir_ = '/mnt/YuHe/work_vechicle/original/work_vechicle/dataset/huzhou/val_data/calc_map_huzhou_val_data/label_txt'
    float_to_int(txt_dir_, img_dir_, save_dir_)
