"""用于提供 ai 接口"""
import traceback
from flask import Blueprint, request, jsonify

from utils.img_utils import decode_image

ai_configuration = Blueprint('ai-configuration', __name__)


@ai_configuration.route('/AIPredict', methods=['POST', 'GET'])
def face_recognize():
    try:
        json_file = request.get_json()  # 读取网页端传入的数据
        image = decode_image(json_file['image'])  # 解码图像
        # TODO: 使用 AI 模型获取推理的结果
        result = {}  # 返回最终的结果
        # return jsonify(result)
        return "login fail", 404, [("token", "123456"), ("City", "shenzhen")]
    except Exception as e:
        traceback.print_exc()
        return {"error": e}
