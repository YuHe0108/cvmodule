"""数据据拆分: """

from collections import defaultdict
import xml.etree.ElementTree as ET
import argparse
import shutil
import random
import json
import yaml
import sys
import os

TOTAL_CNT = defaultdict(int)
MISS = defaultdict(int)
MISS_LABEL = set()

LABEL_PATH = None


def parse_args(args):
    # 处理参数
    parser = argparse.ArgumentParser(description="split dataset.")
    parser.add_argument("--data", help="*.yaml 制定data文件夹里的yaml配置文件", required=False, type=str,
                        default='/mnt/YuHe/data/SDYD/person_waste/data_left_xj3.yaml')
    parser.add_argument("--label", help="标签的路径", required=False, type=str,
                        default='/mnt/YuHe/data/SDYD/person_waste/label.json')
    parser.add_argument("--split_data", default=0.1, help="需要加在的数据路径", required=False, type=str)
    parser.add_argument("--valid_radio", default=0.1, help="valid data set radio", required=False, type=float)
    parser.add_argument("--test_radio", default=0.0, help="test data set radio", required=False, type=float)
    return parser.parse_args(args)


def get_labels():
    global LABEL_PATH

    if LABEL_PATH is not None and not os.path.exists(str(LABEL_PATH)):
        return []

    with open(str(LABEL_PATH), 'r') as file:
        names = json.load(file)
    return names


def clean_files(data_dir):
    # 清除文件
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print("clean dir:{} success".format(data_dir))
    return


def mkdir_and_get(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def label_index(label, labels):
    label = label.lower()
    names = get_labels()
    if label in names:
        TOTAL_CNT[label] += 1
        return names[label]
    MISS[label] += 1
    return -1


def write_content_to_file(content, file_path):
    with open(file_path, 'w') as f:
        f.write(content)


def write_line_to_file(line, file_path):
    with open(file_path, 'a') as f:
        f.writelines(line + "\n")


def generate_txt(xml_path, txt_path, labels):
    global MISS_LABEL

    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.find('size')[0].text)
    height = int(root.find('size')[1].text)
    write_content_to_file("", txt_path)
    if len(root.findall('object')) == 0:
        return

    coord_idx = 4
    is_no_target = True
    for member in root.findall('object'):
        try:
            xmin = int(member[coord_idx][0].text)
            ymin = int(member[coord_idx][1].text)
            xmax = int(member[coord_idx][2].text)
            ymax = int(member[coord_idx][3].text)

            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = xmax if xmax < width else width
            ymax = ymax if ymax < height else height

            xc, yc, w, h = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height, (xmax - xmin) / width, (ymax - ymin) / height
            label = member[0].text
            index = label_index(label, labels)
            if index < 0:
                if label not in MISS_LABEL:
                    MISS_LABEL.add(label)
                    print("MISSING LABEL: ", label)
                continue
            is_no_target = False
            line = " ".join([str(index), str(xc), str(yc), str(w), str(h)])
            write_line_to_file(line, txt_path)
        except Exception as e:
            print("error:{}".format(xml_path), e)
            break

    if is_no_target and os.path.exists(txt_path):
        os.remove(txt_path)
    return is_no_target


def main(args):
    global LABEL_PATH

    args = parse_args(args)
    config_yaml = args.data
    test_radio = args.test_radio
    valid_radio = args.valid_radio
    LABEL_PATH = args.label
    split_data = args.split_data

    # with open(r'./data/{}'.format(config_yaml), 'r') as f:
    with open(config_yaml, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    print(conf)

    if len(split_data) != 0 and os.path.exists(split_data):
        dataset_path = split_data
    else:
        dataset_path = conf.get("path")
    print(dataset_path)

    raw_path = os.path.join(dataset_path, "raw")
    images_path = mkdir_and_get(os.path.join(dataset_path, "images"))
    lables_path = mkdir_and_get(os.path.join(dataset_path, "labels"))

    train_path = mkdir_and_get(os.path.join(dataset_path, conf.get("train")))

    print("train_path:{}".format(train_path))

    train_label_path = mkdir_and_get(train_path.replace("images/", "labels/"))
    test_path = mkdir_and_get(os.path.join(dataset_path, conf.get("test")))
    test_label_path = mkdir_and_get(test_path.replace("images/", "labels/"))
    valid_path = mkdir_and_get(os.path.join(dataset_path, conf.get("val")))
    valid_label_path = mkdir_and_get(valid_path.replace("images/", "labels/"))
    labels = conf.get("names")

    print("define labels num:{}, labels:{}".format(len(labels), labels))

    xml_files = []
    for path, dir_list, file_list in os.walk(raw_path):
        print(path)
        for xml_file in file_list:

            if not xml_file.endswith(".xml"):
                # 不是xml 不处理
                continue
            try:
                tree = ET.parse(os.path.join(path, xml_file))
                root = tree.getroot()
                if len(root.findall('object')) == 0:
                    # 没有标签不处理
                    # if random.randint(0,5) == 1:
                    xml_files.append(xml_file)
                    continue
                has_label = False
                for member in root.findall('object'):
                    label = member[0].text
                    # if labels.count(label) > 0:
                    #     for b in target_labels:
                    #         if b in label.lower():
                    has_label = True
                    break
                if has_label:
                    xml_files.append(xml_file)
            except Exception as e:
                print(xml_file, e)

    num = len(xml_files)

    valid_test_list = random.sample(xml_files, int(num * (test_radio + valid_radio)))
    train_list = list(set(xml_files) - set(valid_test_list))

    valid_test_num = len(valid_test_list)

    test_list = random.sample(valid_test_list, int(valid_test_num * (test_radio / (test_radio + valid_radio))))

    valid_list = list(set(valid_test_list) - set(test_list))

    print("dataset total num:{},train dataset num:{},test dataset num:{},valid dataset num:{}".format(num,
                                                                                                      len(train_list),
                                                                                                      len(test_list),
                                                                                                      len(valid_list)))

    clean_files(test_path)
    clean_files(test_label_path)
    clean_files(valid_path)
    clean_files(valid_label_path)
    clean_files(train_path)
    clean_files(train_label_path)

    error_count = 0

    img_suffix = 'jpg'
    for test_file in test_list:
        test_xml = os.path.join(raw_path, test_file)
        test_txt = os.path.join(test_label_path, os.path.splitext(test_file)[0] + ".txt")
        for suffix in ['png', 'jpg']:
            test_image = os.path.join(raw_path, os.path.splitext(test_file)[0] + f".{suffix}")
            if os.path.exists(test_image):
                break

        if not os.path.exists(test_image):
            error_count += 1
            continue
        is_no_target = generate_txt(test_xml, test_txt, labels)
        if not is_no_target:
            shutil.copy(test_image, test_path)

    for valid_file in valid_list:
        valid_xml = os.path.join(raw_path, valid_file)
        valid_txt = os.path.join(valid_label_path, os.path.splitext(valid_file)[0] + ".txt")
        valid_image = os.path.join(raw_path, os.path.splitext(valid_file)[0] + f".{img_suffix}")
        if not os.path.exists(valid_image):
            error_count += 1
            continue

        is_no_target = generate_txt(valid_xml, valid_txt, labels)
        if not is_no_target:
            shutil.copy(valid_image, valid_path)

    for train_file in train_list:
        train_xml = os.path.join(raw_path, train_file)
        train_txt = os.path.join(train_label_path, os.path.splitext(train_file)[0] + ".txt")
        train_image = os.path.join(raw_path, os.path.splitext(train_file)[0] + f".{img_suffix}")

        if not os.path.exists(train_image):
            error_count += 1
            continue
        is_no_target = generate_txt(train_xml, train_txt, labels)
        if not is_no_target:  # NEW: 如果图像中不存在目标，则不将此数据放入训练集和测试集中
            shutil.copy(train_image, train_path)

    print("success split test dataset and train dataset!!!,has not image num:{}".format(error_count))
    print(f"MISS LABEL: {MISS}")
    print(f"TOTAL COUNT: {TOTAL_CNT}")
    return


if __name__ == '__main__':
    main(sys.argv[1:])
