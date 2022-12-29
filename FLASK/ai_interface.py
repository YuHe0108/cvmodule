"""用于提供 ai 接口"""
import os
import time
import uuid
import queue
import sqlite3
import threading

import matplotlib.pyplot as plt
import requests
import traceback
from flask import Blueprint, request, jsonify

from FLASK.data_persist import DataPersist
from utils.img_utils import decode_image, download_img

ai_configuration = Blueprint('ai-configuration', __name__)

DATA_BASE = DataPersist("interface.db")  # 创建数据库
IMAGE_SAVE_DIR = "cache_images"
result = queue.Queue()


def process():
    global result

    for i in range(10):
        time.sleep(1)
        result.put_nowait(i)
        print(i)
    return


@ai_configuration.route('/ai_predict', methods=['POST', 'GET'])
def ai_predict():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        image = decode_image(json_file['image'])  # 解码图像
        # TODO: 使用 AI 模型获取推理的结果
        res = {}  # 返回最终的结果
        # return jsonify(res)
        return {"ok": 1}
    except Exception as e:
        traceback.print_exc()
        return {"error": e}


@ai_configuration.route('/download_file', methods=['GET', 'POST'])
def download_file():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        if "download_url" not in json_file:
            return {"error": "download_url not input"}
        if "file_name" not in json_file:
            return {"error": "file_name not input"}
        if "save_name" not in json_file:
            return {"error": "save_name not input"}
        download_url = json_file['download_url']
        file_name = json_file['file_name']
        save_name = json_file['save_name']
        file_url = "{}?id={}".format(download_url, file_name)
        download_img(file_url, f'{save_name}.jpg')
        return {"ok": 1}
    except Exception as e:
        traceback.print_exc()
        return {"error": e}


@ai_configuration.route('/update_config', methods=['GET'])
def update_config():
    try:
        config = {}
        update_config_info = {"update_config_url": "10.10.10.158"}
        task_config_info = {"task_name": "reload_image",
                            "root_dir": "/mnt/YuHe",
                            "save_dir": "/mnt/YuHe/data"}

        config["base"] = update_config_info
        config["task"] = task_config_info
        return config
    except Exception as e:
        traceback.print_exc()
        return {}


@ai_configuration.route('/test', methods=['POST', 'GET'])
def test():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        print(json_file)
        return {"ok": 1}
    except Exception as e:
        traceback.print_exc()
        return {"error": e}


@ai_configuration.route('/asynchronous_test', methods=['POST', 'GET'])
def asynchronous_test():
    global IMAGE_SAVE_DIR
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        image = decode_image(json_file["image"])
        cache_name = uuid.uuid1().hex
        plt.imsave(os.path.join(IMAGE_SAVE_DIR, f"{cache_name}.jpg"), image)
        DATA_BASE.insert_data("ai_interface",
                              {"createTime": time.time(), "image": str(cache_name), "result": 22})
        t1 = threading.Thread(target=process, args=())
        t1.start()
        # 将此记录在数据库中, 在结果列设置为0
        return {"ok": 1}
    except Exception as e:
        traceback.print_exc()
        return {"error": e}


@ai_configuration.route('/asynchronous_result', methods=['POST', 'GET'])
def asynchronous_result():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        t = time.time()
        res = DATA_BASE.get_data("ai_interface")
        if len(res) > 0:
            DATA_BASE.insert_data("upload_completed", res)  # 移动到完成列表
            DATA_BASE.remove_top_nums_data("ai_interface", 1)  # 删除原数据
        print(time.time() - t)
        return {"ok": res}
    except Exception as e:
        traceback.print_exc()
        return {"error": e}
