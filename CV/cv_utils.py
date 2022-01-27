import os
import time
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from cvmodule import utils


def inverse_image(image):
    if type(image) is str:
        image = cv.imread(image)
    height, width, channels = image.shape

    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    return image


def binary_image(image, threshold):
    """使用固定的阈值 threshold 二值化图像"""
    if type(image) is str:
        image = cv.imread(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    th, bin_img = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return bin_img


def pic_to_video(img_dir, size):
    """将多张图像合成为一个视频"""
    img_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
    fps = 2
    video_path = os.path.join(img_dir, 'synthesis.avi')
    fourcc = cv.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv.VideoWriter(video_path, fourcc, fps, size)

    for path in img_paths:
        img = cv.imread(path)
        if img is None:
            img = plt.imread(path)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.resize(img, size)
        video.write(img)

    video.release()
    return


def get_file_path(file_dir):
    path_list = []
    for path in pathlib.Path(file_dir).iterdir():
        str_path = str(path)
        if os.path.isfile(str_path):
            path_list.append(str_path)
    return sorted(path_list, key=lambda path: int(pathlib.Path(path).stem))


def crop_image(read_dir, save_dir, o_w, o_h, r_w, r_h, split=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将大幅图像裁剪为小图像
    file_paths_list = get_file_path(read_dir)

    Img_W = o_w // r_w
    Img_H = o_h // r_h

    i = 0
    for file_path in file_paths_list:
        for row in range(Img_H):
            for col in range(Img_W):
                img = cv.imread(file_path)
                croped = img[row * r_h: (row + 1) * r_h, col * r_w: (col + 1) * r_w, :]

                if split:
                    save_path = os.path.join(save_dir, str(pathlib.Path(file_path).stem))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv.imwrite(os.path.join(save_path, '{0}.jpg'.format(i)), croped)
                else:
                    cv.imwrite(os.path.join(save_dir, '{0}.jpg'.format(i)), croped)
                i += 1
    print('处理完成!')
    return 0


def rename_file(img_dir, save_dir):
    path_list = get_file_path(img_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index, path in enumerate(path_list):
        img = cv.imread(path)
        cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(index)), img)
    return


def draw_contours_by_mask(image_dir,
                          mask_dir,
                          save_dir,
                          mask_prefix=False,
                          reshape_shape=(1440, 1440),
                          max_count=8,
                          color=(255, 0, 0)):
    """将mask的轮廓绘制在原始图像上"""
    utils.check_file(save_dir)
    images_path, images_stem = utils.get_path_and_stem(image_dir)
    masks_path, masks_stem = utils.get_path_and_stem(mask_dir)

    for stem in images_stem:
        if mask_prefix:
            index = masks_stem.index('mask_{}.png'.format(stem[:-4]))
        else:
            index = masks_stem.index(stem)
        # 分别根据文件的stem读取图像和mask
        image_path = images_path[index]
        original_image = cv.imread(str(image_path))
        original_image = cv.resize(original_image, reshape_shape)

        mask_path = masks_path[index]
        mask_image = cv.imread(mask_path, 0)
        mask_image = cv.resize(mask_image, reshape_shape)
        threshold, bin_mask_image = cv.threshold(mask_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # 寻找bin_mask_image中的轮廓
        con_list = []
        contours, _ = cv.findContours(bin_mask_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            # 根据面积的大小过滤一部分轮廓
            area = cv.contourArea(contour)
            if 300 < area < reshape_shape[0] * reshape_shape[1] * 0.40:
                con_list.append(index)

        # 一张图中最多绘制多少个轮廓
        drawed_image = original_image
        if len(con_list) > max_count:
            drawed_image = np.zeros(shape=reshape_shape, dtype=np.uint8)
        else:
            for index in con_list:
                drawed_image = cv.drawContours(drawed_image, contours, index, color, 5)

        cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(stem)), drawed_image)
    return


if __name__ == '__main__':
    # pic_to_video(r'E:\tookit_backup\DL_Projects\Projects\目标检测\EfficientDet\inval_images', (512, 512))
    # crop_image(r'D:\Users\YingYing\Desktop\dwnet_invalid_pred_images',
    #            r'D:\Users\YingYing\Desktop\dwnet_invalid_pred_images',
    #            256 * 4, 256 * 2, 256, 256)
    rename_file(r'D:\Users\YingYing\Desktop\test_mask',
                r'D:\Users\YingYing\Desktop\rename_mask')
