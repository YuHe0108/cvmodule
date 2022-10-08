"""numba的使用方法"""
from numba import cuda
import numpy as np
import numba
import time


@numba.jit(nopython=True)
def cal_sum(a):
    result = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result += a[i, j]
    return result


@cuda.jit
def gpu_dot_sum(a, b, result, n, c=None):
    """采用GPU实现矩阵的对应点相乘再相加"""
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        row, col = a[idx].shape
        temp = 0
        for i in range(row):
            for j in range(col):
                if c is None:
                    temp += a[idx][i, j] * b[i, j]
                else:
                    temp += a[idx][i, j] * b[i, j] * c[i, j]
        result[idx] = temp
    return


def calc_background_mean(parts):
    total = len(parts)
    # ring_mat = generate_ring_mat(parts[0])  # 生成圆环区域
    ring_mat = []
    mean_ring_kernel = np.ones(shape=ring_mat.shape) * ring_mat * (1 / ring_mat.size)
    # 移动至 GPU 的显存中
    parts_device = cuda.to_device(parts)
    ring_mat_device = cuda.to_device(ring_mat)
    mean_ring_kernel_device = cuda.to_device(mean_ring_kernel)
    res = cuda.device_array(total)
    gpu_dot_sum(parts_device, ring_mat_device, res, mean_ring_kernel_device)  # 计算结果
    return res.copy_to_host()  # 将结果返回值 CPU


@cuda.jit(device=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)
    return


if __name__ == '__main__':
    # value = np.random.random((5000, 5000))
    # t = time.time()
    # cal_sum(value)
    # print(time.time() - t)
    for _ in range(2):
        gimage = np.zeros((1024, 1536), dtype=np.uint8)
        blockdim = (32, 8)
        griddim = (32, 16)

        start = time.time()
        d_image = cuda.to_device(gimage)
        mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20)
        res = d_image.copy_to_host()
        dt = time.time() - start
        print(dt)
