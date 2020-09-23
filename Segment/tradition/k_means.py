import matplotlib.pyplot as plt
from tf_package import utils
import numpy as np
import cv2 as cv
import pathlib
import tqdm
import os


def k_means_fun(inputs, save_dir=None, k_cluster=3, threshold=160):
    utils.check_file(save_dir)
    # 判断输入类别
    all_image_paths = utils.return_inputs(inputs)
    all_image_paths = utils.is_same_two_path_list(inputs, save_dir)

    for path in tqdm.tqdm(all_image_paths, total=len(all_image_paths)):
        stem = pathlib.Path(path).stem
        original_image = cv.imread(path)
        img = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        vectorized = img.reshape((-1, 1))
        # convert to np.float32
        vectorized = np.float32(vectorized)
        # Here we are applying k-means clustering so that the pixels around a colour are consistent and gave same BGR/HSV values
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # We are going to cluster with k = 2, because the image will have just two colours ,a white background and the colour of the patch
        attempts = 10
        ret, label, center = cv.kmeans(vectorized, k_cluster, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
        # Now convert back into uint8
        # now we have to access the labels to regenerate the clustered image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        th, bin_res = cv.threshold(res2, threshold, 255, cv.THRESH_BINARY)
        if save_dir is not None:
            cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(stem)), bin_res)
    return


if __name__ == '__main__':
    k_means_fun(r'D:\Users\YingYing\Desktop\data\no_skull')
