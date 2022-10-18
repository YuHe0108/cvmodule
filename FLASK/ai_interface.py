"""用于提供 ai 接口"""
import requests
import traceback
from flask import Blueprint, request, jsonify

from utils.img_utils import decode_image, download_img

ai_configuration = Blueprint('ai-configuration', __name__)


@ai_configuration.route('/ai_predict', methods=['POST', 'GET'])
def ai_predict():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        image = decode_image(json_file['image'])  # 解码图像
        # TODO: 使用 AI 模型获取推理的结果
        result = {}  # 返回最终的结果
        # return jsonify(result)
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
