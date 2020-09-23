import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)


def fcm(img_path, num_cluster, b, reshape_size=None):
    """
    :param img_path: 输入图像地址
    :param num_cluster: 聚成几类
    :param b:
    :param reshape_size: reshape图像尺寸
    :return:
    """
    img = cv.imread(img_path)
    if reshape_size:
        img = cv.resize(img, reshape_size)
    h, w, c = img.shape
    img = img.astype(np.double)
    inputs = img.reshape(h * w, c)  # 二维矩阵
    N, c = inputs.shape

    P = np.random.randn(N, num_cluster)  # Generating Random Numbers with Normal Distribution
    P = P / np.dot(np.sum(P, 1).reshape(N, 1), np.ones((1, num_cluster)))

    J_prev = np.inf
    J = []
    iterations = 0
    while True:
        t = pow(P, b)
        C = np.dot(inputs.T, t).T / (sum(t, 0).reshape(num_cluster, 1) * np.ones((1, c)))
        dist = np.dot(np.sum(C * C, 1).reshape(num_cluster, 1), np.ones((1, N))).T + np.sum(
            inputs * inputs, 1).reshape(N, 1) * np.ones((1, num_cluster)) - 2 * np.dot(inputs, C.T)

        t2 = pow(1.0 / dist, 1.0 / (b - 1))
        P = t2 / (np.sum(t2, 1).reshape(N, 1) * np.ones((1, num_cluster)))
        J_cur = sum(sum((pow(P, b)) * dist, 0), 0) / N
        J.append(J_cur)

        print(iterations, J_cur)
        if abs(J_cur - J_prev) < 0.001:
            break

        iterations += 1
        J_prev = J_cur
    label = dist.argmin(axis=1)
    img_1 = C[label, :]
    result = img_1.reshape(h, w, c)
    # plt.imshow(result / 255, cmap='gray')
    # plt.show()
    return result


if __name__ == '__main__':
    fcm('brain.jpg', 3, 2, (256, 256))
