#!/bin/bash
# shellcheck disable=SC2164
# shellcheck disable=SC2006

# 激活环境
conda env list
source activate pytorch-1.8-py37

function create_folder() {
  folder_path=$1
  temp_folder_path=$1
  for i in {1..20}
  do
    if [ ! -d "$temp_folder_path" ]; then
      mkdir "$temp_folder_path"
      echo "$temp_folder_path"
      break
    elif [ ! "$(ls -A "$temp_folder_path")" ]; then
      echo "$temp_folder_path"
      break
    else
      temp_folder_path="$folder_path-$i"
    fi
  done
}

# 新数据准备
data_root=$1
if [ ! -d "$data_root" ]; then
  echo "$data_root 文件夹不存在!"
  exit
fi
current_date="`date +%Y-%m-%d`"

# 通过shell输入参数
batch_size=$2
device=$3
if [ ! "$batch_size" ]; then
  echo "batch_size 为空!"
  exit
fi
if [ !  "$device" ]; then
  echo "device 为空!"
  exit
fi

# 图像重新读取并加载
img_root="$data_root/raw"
reload_img_save_path="$data_root/reload_images"
cd "/mnt/YuHe/python script/"
python reload_image_2.py --root "${img_root}" --save_root "${reload_img_save_path}"
python copy_files.py --root "${reload_img_save_path}" --save_root "${img_root}"

# yolo-v5 数据准备
cd "/mnt/YuHe/work_vechicle/original/work_vechicle"
label_path="/mnt/YuHe/data/SDYD/left/left-label.json"
data_path="/mnt/YuHe/data/SDYD/left/detection/data_left.yaml"
python split_data.py  --split_data "${data_root}" --label $label_path --data $data_path --test_radio 0.0 --valid_radio 0.1
save_root="/mnt/YuHe/data/SDYD/left/detection/train_val_data"
#save_root="/mnt/YuHe/backup/train_val_data"
cp_root="/mnt/YuHe/data/SDYD/left/detection/train_val_data_copy_$current_date"
#cp_root="/mnt/YuHe/backup/train_val_data_$current_date"
res=$(create_folder "$cp_root") # 文件夹创建
cp_root=$res
echo "$cp_root"

# 为防止数据丢失, 文件夹复制
cd "/mnt/YuHe/python script/"
python copy_directory.py --ori_root "${save_root}" --save_root "${cp_root}"
# 文件过滤, 如果save_root中存在，文件名一致，且大小相同的文件，则移除
python remove_exist_file.py --img_root "${save_root}/images/train" --txt_root "${save_root}/labels/train" --target_root "${img_root}" --save_root "${save_root}/repeat_file"
python remove_exist_file.py --img_root "${save_root}/images/val" --txt_root "${save_root}/labels/val" --target_root "${img_root}" --save_root "${save_root}/repeat_file"
python remove_exist_file.py --img_root "${save_root}/raw" --txt_root "${save_root}/raw" --target_root "${img_root}" --save_root "${save_root}/repeat_file"
cp -r "${data_root}/images/" $save_root
cp -r "${data_root}/labels/" $save_root
cp -r "${data_root}/raw/" $save_root

# 将测试集复制一份至验证下
save_valid_path="/mnt/YuHe/data/val_data/left/detection/history"
#save_valid_path="/mnt/YuHe/backup"
new_folder_path=${save_valid_path}/"`date +%Y-%m-%d`"
res=$(create_folder "$new_folder_path") # 文件夹创建
new_folder_path=$res
echo "$new_folder_path"

val_path="$data_root/images/val"
cp -r "$val_path" "$new_folder_path"
mv "$new_folder_path/val" "$new_folder_path/raw"

cd "/mnt/YuHe/python script/"
python find_crosspond_file.py --img_root "$new_folder_path/raw" --xml_dir "$data_root/raw"

# 将新的验证集移动到指定文件夹下
valid_root="/mnt/YuHe/data/val_data/left/detection/cur-val-data-detection"
# valid_root="/mnt/YuHe/backup/val"
# TODO: 对于 valid_root, 如果其中的文件在img_root中，就先移除
python remove_exist_file.py --img_root "${valid_root}" --txt_root "${valid_root}" --target_root "${img_root}" --save_root "/mnt/YuHe/data/val_data/left/detection/repeat_file"
# python remove_exist_file.py --img_root "${valid_root}" --txt_root "${valid_root}" --target_root "${img_root}" --save_root "/mnt/YuHe/backup/repeat_file"
python copy_files.py --root "$new_folder_path/raw" --save_root $valid_root
echo "验证集准备完成!"

# 训练模型
cd /mnt/YuHe/work_vechicle/ground_line/yolov5-xj3
python train_left_detect.py --batch-size "$batch_size" --device "$device"
