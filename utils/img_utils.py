from PIL import Image
import numpy as np
import requests
import hashlib
import base64
import cv2
import io


def image_process(img, value_range, input_format='channel_first'):
    """ 将图像归一化至 01（-1, 1）之间，并转换 HWC ~ CHW, 增加维度： BCHW"""
    inputs = img.astype(np.float32)
    if value_range == '01':  # 将 img 的值归一化至 0-1 之间
        inputs /= 255
    else:  # -1 ~ 1 之间
        inputs = inputs / 127.5 - 1

    if input_format == 'channel_first':
        inputs = np.transpose(inputs, [2, 0, 1])
    return np.expand_dims(inputs, 0)


def join_img(img, shape):
    """将图像通过拼接的方式增大图像尺寸"""
    out = img
    h, w = out.shape[:2]
    ori_h, ori_w = img.shape[:2]
    while h < shape[0] or w < shape[1]:
        out = cv2.copyMakeBorder(out, ori_h, 0, ori_w, 0, cv2.BORDER_WRAP)
        h, w = out.shape[:2]
    return out[0:shape[0], 0:shape[1]]


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """缩放图像尺寸至： new_shape"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def draw_img(img, box, color, int_2_label, suffix=''):
    for x1, y1, x2, y2, cls in box:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if color == (0, 0, 255):
            cv2.putText(img, int_2_label[int(cls)] + "-" + suffix,
                        (x1, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        else:
            cv2.putText(img, int_2_label[int(cls)] + "-" + suffix,
                        (x1, y2 - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return img


def decode_image(encoding_image):
    """通过 json 文件解析 image, 将 base64 转为 rgb"""
    image = base64.b64decode(encoding_image)
    image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    image = np.array(image, dtype=np.uint8)  # 将图像转换为 array 形式
    return image


def download_img(url, save_path):
    headers_ = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers_, timeout=5, verify=False)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    response.close()
    return


if __name__ == '__main__':
    # img = cv2.imread(r'D:\Download\5.jpg')
    # img_h, img_w = img.shape[:2]
    # box = [[0.7528735632183908, 0.3911290322580645, 0.09051724137931035, 0.3870967741935484]]
    # for x, y, w, h in box:
    #     x = int(x * img_w)
    #     y = int(y * img_h)
    #     x2 = x + int(w * img_w)
    #     y2 = y + int(h * img_h)
    #     cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    files = ['1853c277d51045e1af0f3a64a6af4c1b', '6b591e70726f44c69a64620121927a2c', '99611749d4e143b796dbf740caa2851f',
             '83122077e83d433c9db2bfad7505ae34', '55975b97915b41278c6c0fe4798f9fc3', 'c3d6e1aa5f5c43b782fedb7bb56bf67a',
             '642b7640898341f2959aded17433ba2e', '3efb85c311744b4d83601a34cfe36810', 'd57da847ae6245f59bc7fc76a3e4e750',
             'c8f6e6c9aa1e43f0adcda6ed8d90d3fb', '2159651a4aeb4c948b5588ed79294b5f', 'fc87a92873f0480b98453bacd51a9ba4',
             '2ffcafcb79cd4b8689ef814f8e40e087', 'f98c540e03394531a16b0050ec879bba', '5db55f96955841dca9e71235fc146555',
             '3d3fdddd9f7a4b25929363c75ff95cd0', 'f4b16e9503cc4417af3d1a24fb87e250', '7e3f88a13b4b44c092be52755121f99f',
             '1ea6009178d54cd5a8422ff7f7c9df1c', 'c0af97e8e5dc4755a7ea089794108a46', 'ae79bb09a7c54603a4011fd8b40793cc',
             '9709a64f84c34be78e1efd64457a66fa']
    base = 30
    for file in files:
        download_img("{}?id={}".format('http://222.92.212.123:8003/cloudFile/common/downloadFile', file),
                     f'{base}.jpg')
        base += 1
