from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pathlib
import torch
import json
import tqdm
import glob
import sys
import os

sys.path.insert(0, "..\\Detection\\yolov5")
sys.path.insert(0, "..\\")

import tools
from Detection.yolov5.utils.datasets import letterbox
from Detection.yolov5.utils.general import non_max_suppression, scale_coords
from Detection.utils import visualization as vis_util
from Detection.utils import calc

"""计算模型的 map """


class CalcMAP:
    def __init__(self, weight_path, label_file, img_shape, data_dir=None, img_dir=None, xml_dir=None):
        """
        img_shape:      测试 resize 图像尺寸
        label_file:     标签文件，txt 格式即可
        data_dir:       图像和 xml 文件放在一起，则只提供当前路径即可
        img_dir:        验证集图像的路径
        xml_dir:        对应图像 xml 标签的路径
        """
        assert data_dir is not None or (img_dir is not None and xml_dir is not None)
        if data_dir is not None:
            assert os.path.exists(data_dir)
        if img_dir is not None and xml_dir is not None:
            assert os.path.exists(img_dir) and os.path.exists(xml_dir)

        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.data_dir = data_dir
        self.label_file = label_file
        self.input_shape = img_shape
        self.weight_path = weight_path

        self.cover_detect_results = False  # 已经存在的预测结果是否覆盖

        self.root_dir = 'run_map_results'  # 将 预测结果、计算 map 产生的结果统一存放的路径
        self.pred_dir = None  # 预测结果 txt 文件的保存位置
        self.label_txt_dir = None  # txt 标签抓换后的存放路径
        self.save_calc_results_dir = None  # 计算 map 保存结果的路径
        self.draw_pred_dir = None  # 预测结果在原图绘制的结果
        self.json_file_dir = os.path.join(self.root_dir, 'json_file')  # 将 label 标签重新保存为 json 格式 的路径

        self.show_animation = False  # 显示保存预测的结果
        self.draw_plot = True  # 绘制结果图
        self.specific_iou_classes = {}  # 是否为每一个类别独立设置 IOU
        self.specific_iou_flagged = False if self.specific_iou_classes is None else True

        self.min_overlap = 0.5
        self.min_score_thresh = 0.5  # 置信度大于该值时，认为是目标
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        self.idx2label = self.read_label()  # 标签有哪些
        self.label2idx = {label: i for i, label in self.idx2label.items()}
        self.ignore_labels = set()  # 哪些类不参与 map 计算，自行添加
        self.img_suffix = {'.jpg', 'jpeg', 'JPG', 'JPEG', 'png'}  # 数据集图像的 后缀有哪些

    def calculate(self):
        self.xml2txt()  # 首先将 xml 文件转换为 txt
        self.pred()  # 对每张图预测推理，并将推理结果保存至本地 txt 文档
        self.calc_map()  # 根据预测结果和推理结果计算 map
        return

    def xml2txt(self):
        """将 xml 文件转换成 txt 文件, 并保存在与 xml 相同的路径下"""
        xml_dir = self.xml_dir if self.xml_dir is not None else self.data_dir
        self.label_txt_dir = xml_dir
        for xml_file in pathlib.Path(xml_dir).iterdir():
            file_name = pathlib.Path(xml_file).stem
            save_txt_path = os.path.join(xml_dir, "{}.txt".format(file_name))  # 需要保存的txt文件
            if xml_file.suffix != '.xml' or os.path.exists(save_txt_path):
                continue

            # 读取 xml 文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            result_list = []
            for member in root.findall('object'):
                class_name = member[0].text
                idx = self.label2idx.get(class_name, -1)  # 当前不存在于标签中
                if idx < 0:
                    continue
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                msg = "{} {} {} {} {} \n".format(class_name, xmin, ymin, xmax, ymax)
                result_list.append(msg)

            with open(save_txt_path, "w") as f:
                # class_id, conf_score, xmin, ymin, xmax, ymax
                f.writelines(result_list)
        print('xml convert to txt done !')
        return

    def pred(self):
        self.img_dir = self.img_dir if self.img_dir is not None else self.data_dir  # 存放 invalid 图像的路径
        self.pred_dir = os.path.join(self.root_dir, 'predict_txt')  # 预测结果的存放路径
        tools.check_path(self.pred_dir)

        print('开始推理图像....')
        for img_file in tqdm.tqdm(pathlib.Path(self.img_dir).iterdir()):
            if img_file.suffix not in self.img_suffix:
                continue
            img_name = pathlib.Path(img_file).stem  # 预测图像的名字
            # 允许覆盖，并且预测的 txt 结果已经存在，则不在重复计算
            if not self.cover_detect_results and os.path.exists(os.path.join(self.pred_dir, "{}.txt".format(img_name))):
                continue
            inputs, ori_img = self.read_img(str(img_file))  # 读取图像
            boxes, scores, classes = self.inference(inputs, ori_img)  # 通过模型推理出的结果
            self.save_pred_txt(img_name, ori_img, boxes, scores, classes)  # 将预测的结果保存至本地
        print('推理阶段完成！')
        return

    def calc_map(self):
        # self.label_txt_dir
        # self.pred_dir
        self.save_calc_results_dir = os.path.join(self.root_dir, 'calc_results')  # 结果保存的路径
        plot_dir = os.path.join(self.save_calc_results_dir, 'classes')  # 保存 plot 的结果
        self.draw_pred_dir = os.path.join(self.save_calc_results_dir, 'images', 'detections_one_by_one')
        tools.check_path([self.json_file_dir, self.save_calc_results_dir, plot_dir, self.draw_pred_dir])

        gt_counter_per_class = defaultdict(int)  # 记录每个类在总数据集中的数量
        counter_images_per_class = defaultdict(int)  # 记录每个类出现在多少张图像中
        miss_detector_cnt = 0  # 缺少多少预测的 txt 文件

        gt_json_files = []
        # 遍历label标签
        for label_txt_file in pathlib.Path(self.label_txt_dir).iterdir():
            if pathlib.Path(label_txt_file).suffix != '.txt':
                continue

            label_txt_file_name = pathlib.Path(label_txt_file).stem
            pred_txt_file_path = os.path.join(self.pred_dir, f'{label_txt_file_name}.txt')  # 预测的txt路径
            if not os.path.exists(pred_txt_file_path):
                print(f'预测文件不存在：{label_txt_file_name}')
                miss_detector_cnt += 1
                continue

            is_difficult = False
            bounding_boxes = []  # 真实标注框的信息
            already_seen_classes = set()  # 当前图像有相同类出现两个以上时，只算一次
            lines_list = tools.file_lines_to_list(
                os.path.join(self.label_txt_dir, label_txt_file))  # 读取txt文档，并返回列表形式
            class_name = left = top = right = bottom = _difficult = None
            for line in lines_list:
                # 读取 label txt 信息
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                except ValueError:
                    print("预测 txt 文档格式不正确！")

                if class_name in self.ignore_labels:  # 是否放弃计算某些类的 map
                    continue

                bbox = left + " " + top + " " + right + " " + bottom  # 目标预测框信息
                if is_difficult:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

                    gt_counter_per_class[class_name] += 1  # 记录每个类的数量

                    if class_name not in already_seen_classes:  # 记录每个类在出现在多少张图像中
                        already_seen_classes.add(class_name)
                        counter_images_per_class[class_name] += 1

            # 将标签重新保存为 json 格式
            label_json_file = os.path.join(self.json_file_dir, f"{label_txt_file_name}_ground_truth.json")
            gt_json_files.append(label_json_file)
            with open(label_json_file, 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())  # 数据集中存在哪些类
        gt_classes = sorted(gt_classes)  # 排序
        n_classes = len(gt_classes)  # 数据集中一共出现类的种类数量

        # 将 预测的 txt 同样转换为 json 格式
        print('json 文件转换 ....')
        for class_index, class_name in enumerate(gt_classes):  # 按照类别逐个计算
            bounding_boxes = []
            pred_class_name = confidence = left = top = right = bottom = None
            for pred_txt_file in pathlib.Path(self.pred_dir).iterdir():
                pred_txt_file_name = pathlib.Path(pred_txt_file).stem
                label_txt_file = os.path.join(self.label_txt_dir, f'{pred_txt_file_name}.txt')
                if class_index == 0:
                    if not os.path.exists(label_txt_file):
                        print('label txt file not exist', label_txt_file)

                lines = tools.file_lines_to_list(pred_txt_file)
                for line in lines:
                    try:
                        pred_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        print(f'读取模型预测的txt文件错误：{pred_txt_file_name}.')

                    if pred_class_name == class_name:
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_name": pred_txt_file_name, "bbox": bbox})

            # 按照置信度从大到小排序, 并将预测结果保存为json文件，保存的路径于 GT 相同
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(os.path.join(self.json_file_dir, f'{class_name}_dr.json'), 'w') as outfile:
                json.dump(bounding_boxes, outfile)
        print('json 文件转换完成 ！')

        # 当预测结果与标签全部保存为 json 格式后，开始计算 map
        sum_ap = 0.0
        ap_dictionary = {}  # 每个类别对应一个map值
        lamr_dictionary = {}
        with open(os.path.join(self.save_calc_results_dir, "output.txt"), 'w') as output_file:
            # 将结果输出至 output.txt 文档
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0

                # 加载模型的预测结果
                dr_file = os.path.join(self.json_file_dir, f'{class_name}_dr.json')
                dr_data = json.load(open(dr_file))

                # 将预测值 和 GT 进行匹配
                nd = len(dr_data)
                tp, fp = [0] * nd, [0] * nd
                ground_truth_img = ''
                for idx, detection in enumerate(dr_data):
                    file_name = detection["file_name"]
                    bb = [float(x) for x in detection["bbox"].split()]  # 模型预测的 obj-box
                    if self.show_animation:  # 是否显示
                        # find ground truth image
                        ground_truth_img = glob.glob1(self.img_dir, file_name + ".*")
                        ground_truth_img = [path for path in ground_truth_img if
                                            pathlib.Path(path).suffix in self.img_suffix]

                        if len(ground_truth_img) == 0:
                            print("Error. Image not found with id: " + file_name)
                            exit()
                        else:
                            img = cv.imread(os.path.join(self.img_dir, str(ground_truth_img[0])))
                            img_cumulative = img.copy()
                            bottom_border = 60  # 在图像的底部添加 border
                            img = cv.copyMakeBorder(img, 0, bottom_border, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])

                    # 读取 gt 的 json 文件
                    gt_file = os.path.join(self.json_file_dir, f"{file_name}_ground_truth.json")
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = gt_match = -1
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:  # 模型预测和标签的类别一致，计算
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                                  min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:  # 计算二者的 IOU
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (
                                        bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    if self.show_animation:
                        status = "NO MATCH FOUND!"  # status is only used in the animation
                    if self.specific_iou_flagged:
                        if class_name in self.specific_iou_classes:
                            self.min_overlap = self.specific_iou_classes[class_name]
                    if ovmax >= self.min_overlap:
                        if "difficult" not in gt_match:  # 标记当前的框已经匹配过
                            if not bool(gt_match["used"]):  # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1

                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                                if self.show_animation:
                                    status = "MATCH!"
                            else:  # false positive (multiple detection)
                                fp[idx] = 1
                                if self.show_animation:
                                    status = "REPEATED MATCH!"
                    else:  # false positive
                        fp[idx] = 1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"

                    # 将预测的标签绘制在图像中
                    if self.show_animation:
                        height, width = img.shape[:2]
                        # colors (OpenCV works with BGR)
                        white = (255, 255, 255)
                        light_blue = (255, 200, 100)
                        green = (0, 255, 0)
                        light_red = (30, 30, 255)
                        # 1st line
                        margin = 10
                        v_pos = int(height - margin - (bottom_border / 2.0))
                        text = "Image: " + ground_truth_img[0] + " "
                        img, line_width = vis_util.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                        img, line_width = vis_util.draw_text_in_image(
                            img, text, (margin + line_width, v_pos), light_blue, line_width)
                        if ovmax != -1:
                            color = light_red
                            if status == "INSUFFICIENT OVERLAP":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(
                                    self.min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(
                                    self.min_overlap * 100)
                                color = green
                            img, _ = vis_util.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                                 line_width)
                        # 2nd line
                        v_pos += int(bottom_border / 2.0)
                        rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                        text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                            float(detection["confidence"]) * 100)
                        img, line_width = vis_util.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result: " + status + " "
                        img, line_width = vis_util.draw_text_in_image(img, text, (margin + line_width, v_pos),
                                                                      color, line_width)

                        font = cv.FONT_HERSHEY_SIMPLEX
                        if ovmax > 0:  # if there is intersections between the bounding-boxes
                            bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                            cv.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font,
                                       0.6, light_blue, 1, cv.LINE_AA)
                        bb = [int(i) for i in bb]
                        cv.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv.putText(img_cumulative, class_name, (bb[0], bb[1] - 5),
                                   font, 0.6, color, 1, cv.LINE_AA)
                        # save image to output
                        output_img_path = os.path.join(self.draw_pred_dir, f"{class_name}_detection{idx}.jpg")
                        cv.imwrite(output_img_path, img)

                # compute precision/recall
                cumsum = 0
                for i, val in enumerate(fp):
                    fp[i] += cumsum
                    cumsum += val
                cumsum = 0
                for i, val in enumerate(tp):
                    tp[i] += cumsum
                    cumsum += val
                rec = tp[:]
                for i, val in enumerate(tp):
                    rec[i] = float(tp[i]) / gt_counter_per_class[class_name]
                prec = tp[:]
                for i, val in enumerate(tp):
                    prec[i] = float(tp[i]) / (fp[i] + tp[i])

                ap, mrec, mprec = calc.voc_ap(rec[:], prec[:])
                sum_ap += ap
                # 输出信息: class_name + " AP = {0:.2f}%".format(ap*100)
                text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
                print(text)
                # 写入至 output.txt 文件中
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec = ['%.2f' % elem for elem in rec]
                output_file.write(
                    text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
                ap_dictionary[class_name] = ap
                n_images = counter_images_per_class[class_name]
                lamr, mr, fppi = calc.log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                lamr_dictionary[class_name] = lamr

                # 将 map 随参数变化的情况绘制出来
                if self.draw_plot:
                    plt.plot(rec, prec, '-o')
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                    fig = plt.gcf()  # gcf - get current figure
                    fig.canvas.set_window_title('AP ' + class_name)
                    plt.title('class: ' + text)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    axes = plt.gca()  # gca - get current axes
                    axes.set_xlim([0.0, 1.0])
                    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                    fig.savefig(os.path.join(self.save_calc_results_dir, 'classes', f"{class_name}.png"))
                    plt.cla()  # clear axes for next plot

            output_file.write("\n# mAP of all classes\n")
            mean_ap = sum_ap / n_classes
            text = "mAP = {0:.2f}%".format(mean_ap * 100)
            output_file.write(text + "\n")
            print(text)
            """
            计算带权重的map
            """
            weight_ap_dictionary = {}
            total_num = 0
            for class_name in sorted(gt_counter_per_class):
                total_num += gt_counter_per_class[class_name]
            weight_mean_ap = 0.0
            for class_name in sorted(gt_counter_per_class):
                class_count = gt_counter_per_class[class_name]
                class_ap = ap_dictionary[class_name]
                weight = class_count / total_num
                weight_ap = weight * class_ap
                weight_mean_ap += weight_ap
                weight_ap_dictionary["{}, num={}".format(class_name, class_count)] = weight_ap
                print("{}:数量={}, 权重={}, AP={},权重AP={}".format(class_name, class_count, weight, class_ap, weight_ap))
            print("权重MAP={}".format(weight_mean_ap))

        if self.draw_plot:
            window_title = "weight-mAP"
            plot_title = "weight-mAP={0:.2f}%,total={1}".format(weight_mean_ap * 100, total_num)
            x_label = "Average Precision"
            output_path = os.path.join(self.save_calc_results_dir, "weight-mAP.png")
            to_show = False
            plot_color = 'royalblue'
            vis_util.draw_plot_func(weight_ap_dictionary, n_classes, window_title, plot_title,
                                    x_label, output_path, to_show, plot_color, "")
        # Draw false negatives
        if self.show_animation:
            pink = (203, 192, 255)
            for tmp_file in gt_json_files:
                ground_truth_data = json.load(open(tmp_file))
                start = self.json_file_dir + '/'
                img_name = tmp_file[tmp_file.find(start) + len(start):tmp_file.rfind('_ground_truth.json')]
                img_cumulative_path = os.path.join(self.save_calc_results_dir, 'images', f"{img_name}.jpg")
                img = cv.imread(img_cumulative_path)
                if img is None:
                    img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
                    img = cv.imread(img_path)
                # draw false negatives
                for obj in ground_truth_data:
                    if not obj['used']:
                        bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                        cv.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
                cv.imwrite(img_cumulative_path, img)

        # Count total of detection-results iterate through all the files
        det_counter_per_class = defaultdict(int)
        for pred_txt_file in pathlib.Path(self.pred_dir).iterdir():
            lines_list = tools.file_lines_to_list(pred_txt_file)  # get lines to list
            for line in lines_list:
                class_name = line.split()[0]
                if class_name in self.ignore_labels:  # check if class is in ignore list,
                    continue
                det_counter_per_class[class_name] += 1

        dr_classes = list(det_counter_per_class.keys())
        # Plot the total number of occurrences of each class in the ground-truth
        if self.draw_plot:
            ground_truth_files_list = glob.glob(self.label_txt_dir + '/*.txt')
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = os.path.join(self.save_calc_results_dir, "ground-truth-info.png")
            to_show = False
            plot_color = 'forestgreen'
            vis_util.draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title,
                                    x_label, output_path, to_show, plot_color, '', )

        # Write number of ground-truth objects per class to results.txt
        with open(os.path.join(self.save_calc_results_dir, 'output.txt'), 'a') as output_file:
            output_file.write("\n# Number of ground-truth objects per class\n")
            for class_name in sorted(gt_counter_per_class):
                output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

        # Finish counting true positives
        for class_name in dr_classes:
            if class_name not in gt_classes:
                count_true_positives[class_name] = 0

        # Plot the total number of occurences of each class in the "detection-results" folder
        if self.draw_plot:
            pred_files_list = glob.glob(self.pred_dir + '/*.txt')
            window_title = "detection-results-info"
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(pred_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            x_label = "Number of objects per class"
            output_path = os.path.join(self.save_calc_results_dir, "detection-results-info.png")
            to_show = False
            plot_color = 'forestgreen'
            true_p_bar = count_true_positives
            vis_util.draw_plot_func(det_counter_per_class, len(det_counter_per_class), window_title,
                                    plot_title, x_label, output_path, to_show, plot_color, true_p_bar)

        # Write number of detected objects per class to output.txt
        with open(os.path.join(self.save_calc_results_dir, 'output.txt'), 'a') as output_file:
            output_file.write("\n# Number of detected objects per class\n")
            for class_name in sorted(dr_classes):
                n_det = det_counter_per_class[class_name]
                text = class_name + ": " + str(n_det)
                text += " (tp:" + str(count_true_positives[class_name]) + ""
                text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
                output_file.write(text)

        if self.draw_plot:
            # Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
            window_title = "lamr"
            plot_title = "log-average miss rate"
            x_label = "log-average miss rate"
            output_path = os.path.join(self.save_calc_results_dir, 'lamr.png')
            to_show = False
            plot_color = 'royalblue'
            vis_util.draw_plot_func(lamr_dictionary, n_classes, window_title,
                                    plot_title, x_label, output_path, to_show, plot_color, "")

            # Draw mAP plot (Show AP's of all classes in decreasing order)
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mean_ap * 100)
            x_label = "Average Precision"
            output_path = os.path.join(self.save_calc_results_dir, "mAP.png")
            to_show = False
            plot_color = 'royalblue'
            vis_util.draw_plot_func(ap_dictionary, n_classes, window_title, plot_title,
                                    x_label, output_path, to_show, plot_color, "")
        print(f'计算完成，一共缺少{miss_detector_cnt}个文件没有参与计算。')
        return

    def get_model(self):
        """ 加载模型 """
        model = torch.load(self.weight_path)['model'].eval().float().to(self.device)  # 加载模型
        return model

    def read_img(self, img_path):
        ori_img = cv.imread(img_path)  # 原始图
        img = letterbox(ori_img, self.input_shape, stride=64, auto=False)[0]  # 缩放后的图
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img, ori_img

    def inference(self, inputs, ori_img):
        # 通过模型得到推理结果
        pred = self.model(torch.tensor(inputs).to(self.device))
        pred = pred[0][None, ...]
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=100)
        ori_img_shape = ori_img.shape
        gn = torch.tensor(ori_img_shape)[[1, 0, 1, 0]]  # [3, 1, 3, 1]

        boxes = []
        scores = []
        classes = []
        for i, det in enumerate(pred):
            # 坐标: scale_coords:
            det[:, :4] = scale_coords(inputs.shape[2:], det[:, :4], ori_img_shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1y1x2y2 = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                boxes.append([x1y1x2y2[1], x1y1x2y2[0], x1y1x2y2[3], x1y1x2y2[2]])
                scores.append(float(conf))
                classes.append(int(cls))

        # 将模型的预测结果绘制在原图中
        draw_res = vis_util.visualize_boxes_and_labels_on_image_array(ori_img, np.array(boxes),
                                                                      np.array(classes), np.array(scores),
                                                                      self.idx2label, use_normalized_coordinates=True,
                                                                      line_thickness=3,
                                                                      min_score_thresh=self.min_score_thresh)
        boxes = np.expand_dims(boxes, 0)
        scores = np.expand_dims(scores, 0)
        classes = np.expand_dims(classes, 0)
        return boxes, scores, classes

    def save_pred_txt(self, img_name, ori_img, boxes, scores, classes):
        height, width = ori_img.shape[:2]
        result_list = []
        for idx, class_id in enumerate(classes[0]):
            conf_score = round(float(scores[0, idx]), 3)  # 置信度保存小数点后三位
            if conf_score > self.min_score_thresh:
                bbox = boxes[0, idx]
                ymin, xmin = int(bbox[0] * height), int(bbox[1] * width)
                ymax, xmax = int(bbox[2] * height), int(bbox[3] * width)
                msg = "{} {} {} {} {} {}\n".format(
                    self.idx2label[int(class_id)], conf_score, xmin, ymin, xmax, ymax)  # 类别、置信度和坐标
                result_list.append(msg)
        with open(os.path.join(self.pred_dir, "{}.txt".format(img_name)), "w") as f:  # 以图像名字为txt名字进行保存
            f.writelines(result_list)  # class_id, conf_score, xmin, ymin, xmax, ymax
        return

    def read_label(self):
        with open(self.label_file, 'r') as file:
            labels = file.readlines()
        idx2label = {i: label.strip() for i, label in enumerate(labels)}
        return idx2label


if __name__ == '__main__':
    WEIGHT_PATH = r'D:\Vortex\Project_7_huzhou\waste_trash_device1280_v1.24.pt'  # 模型权重路径
    DATA_DIR = r'D:\Vortex\Project_7_huzhou\invalid'  # 数据集的路径
    LABEL_FILE = r'D:\Vortex\SELF\cvmodule\Detection\yolov5\data\label.txt'
    calc_map = CalcMAP(WEIGHT_PATH, LABEL_FILE, (1280, 1280), DATA_DIR)
    calc_map.calculate()
