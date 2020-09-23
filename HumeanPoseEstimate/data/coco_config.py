from pathlib import Path
import numpy as np
import json
import time
import os

"""
keypoints: ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'] # 一共有17个关键点
skeleton: [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
           [7, 9], [8, 10],[9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]] # 骨架
"""


class COCO_Keypoints:
    def __init__(self, root_path, txt_save_root):
        self.txt_save_root = txt_save_root  # 保存train和invalid的txt路径
        self.root_path = root_path  # 存放数据集的文件夹路径
        self.annotation = os.path.join(root_path, 'annotations\\')  # 标注文件的路径
        self.train_annotation = os.path.join(self.annotation, "person_keypoints_train2017.json")  # 训练集标注文件
        self.valid_annotation = os.path.join(self.annotation, "person_keypoints_val2017.json")  # 验证集标注文件

    def __print_coco_keypoints(self, data_dict):
        keypoints = data_dict["categories"][0]["keypoints"]  # keypoints
        skeleton = data_dict["categories"][0]["skeleton"]  # 骨架
        print("coco keypoints: {}".format(keypoints))
        print("coco skeleton: {}".format(skeleton))

    @staticmethod
    def __load_json(json_file):
        print("Start loading {}...".format(Path(json_file).name))
        with Path(json_file).open(mode='r') as f:
            load_dict = json.load(f)
        print("Loading is complete!")
        return load_dict

    @staticmethod
    def __get_image_information(data_dict):
        """返回四个列表: 其中包含了训练图像的路径、训练图像的ID、Height、Width"""
        images = data_dict["images"]
        image_file_list = []
        image_id_list = []
        image_height_list = []
        image_width_list = []
        for image in images:
            image_file_list.append(image["file_name"])
            image_id_list.append(image["id"])
            image_height_list.append(image["height"])
            image_width_list.append(image["width"])
        return image_file_list, image_id_list, image_height_list, image_width_list

    def __get_keypoints_information(self, data_dict):
        """图像对应关键点的标注信息: 返回了三个列表，包含了关键点列表、对应图像的ID、BBox"""
        annotations = data_dict["annotations"]
        keypoints_list = []
        image_id_list = []
        bbox_list = []
        for annotation in annotations:
            bbox = annotation["bbox"]
            if self.__is_bbox_valid(bbox):
                keypoints_list.append(annotation["keypoints"])
                image_id_list.append(annotation["image_id"])
                bbox_list.append(bbox)
        return keypoints_list, image_id_list, bbox_list

    @staticmethod
    def __is_bbox_valid(bbox):
        """去除无用的bbox"""
        x, y, w, h = bbox
        if int(w) > 0 and int(h) > 0 and int(x) >= 0 and int(y) >= 0:
            return True
        return False

    @staticmethod
    def __creat_dict_from_list(list_data):
        """将列表制转换为字典: 其中key为列表内容，value是int形数字"""
        created_dict = {}
        for i in range(len(list_data)):
            created_dict[list_data[i]] = i
        return created_dict

    @staticmethod
    def __list_to_str(list_data):
        """将列表制转换为字符串"""
        str_result = ""
        for i in list_data:
            str_result += str(i)
            str_result += " "
        return str_result.strip()

    def __get_the_path_of_picture(self, picture_name):
        return os.path.join(self.images_dir, picture_name)

    """
    One line of txt: xxx.jpg height width xmin ymin w h x1 y1 v1 x2 y2 v2 ... x17 y17 v17
    xxx.jpg：The path of the picture to which the keypoints of the human body belong.
    height: The height of the picture.
    width: The width of the picture.
    xmin: The x coordinate of the upper-left corner of the bounding box. # 左上角的坐标
    ymin: The y coordinate of the upper-left corner of the bounding box.
    w: The width of the bounding box. # bounding box的宽和高
    h: The height of the bounding box.
    xi (i = 1,...,17): The x coordinate of the keypoint.
    yi (i = 1,...,17): The y coordinate of the keypoint.
    vi (i = 1,...,17): When vi is 0, it means that this key point is not marked,
    when vi is 1, it means that this key point is marked but not visible,
    when vi is 2, it means that this key point is marked and also visible.
    """

    def write_information_to_txt(self, dataset):
        """将信息写入txt文件中"""
        if dataset == "train":
            txt_file = os.path.join(self.txt_save_root, 'coco_train_info.txt')
            data_dict = self.__load_json(self.train_annotation)
            self.images_dir = os.path.join(self.root_path, "train2017\\")  # 图像文件路径
        elif dataset == "valid":
            txt_file = os.path.join(self.txt_save_root, 'coco_invalid_info.txt')
            data_dict = self.__load_json(self.valid_annotation)
            self.images_dir = os.path.join(self.root_path, "val2017\\")  # 图像文件路径
        else:
            raise ValueError("Invaid dataset name!")

        # 加载json文件
        self.__print_coco_keypoints(data_dict)  # 打印json信息

        # 返回图像信息
        image_files, image_ids, image_heights, image_widths = self.__get_image_information(data_dict)
        # 返回图像对应的标注信息
        keypoints_list, image_ids_from_keypoints, bboxes = self.__get_keypoints_information(data_dict)
        image_id_dict = self.__creat_dict_from_list(image_ids)
        with open(file=txt_file, mode="a+") as f:
            for i in range(len(image_ids_from_keypoints)):
                one_human_instance_info = ""
                image_index = image_id_dict[image_ids_from_keypoints[i]]
                one_human_instance_info += self.__get_the_path_of_picture(image_files[image_index]) + " "
                one_human_instance_info += str(image_heights[image_index]) + " "
                one_human_instance_info += str(image_widths[image_index]) + " "
                one_human_instance_info += self.__list_to_str(bboxes[i]) + " "
                one_human_instance_info += self.__list_to_str(keypoints_list[i])
                one_human_instance_info = one_human_instance_info.strip()
                one_human_instance_info += "\n"
                print("Writing information of image-{} to {}".format(image_files[image_index], txt_file))
                f.write(one_human_instance_info)
        return


if __name__ == '__main__':
    coco_config = COCO_Keypoints(root_path=r'J:\DATA\ObjDet\COCO',
                                 txt_save_root=r'J:\DATA\ObjDet\COCO')
    coco_config.write_information_to_txt('train')
    # image = cv.imread(r'J:\DATA\ObjDet\COCO\val2017\000000425226.jpg')
    # print(image.shape)
    # drawed_image = cv.rectangle(image, (int(73.35), int(206.02)), (int(73.35 + 300.58), int(206.02 + 372.5)),
    #                             (0, 0, 255), 4)
    # cv.imshow('image', drawed_image)
    # cv.waitKey(0)
