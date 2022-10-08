from numba import cuda, float32
import numpy as np
import time
import math
import cv2


def vec_add(n, a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]


@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x

    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x

    start = tx + ty * block_size
    stride = block_size * grid_size

    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


def test_add():
    n = 20
    x = np.arange(n).astype(np.float32)
    y = 2 * x
    out = np.empty_like(x)

    threads_per_block = 128
    blocks_per_grid = 30

    start = time.time()
    add_kernel[blocks_per_grid, threads_per_block](x, y, out)
    print('gpu cost time is:', time.time() - start)
    print(out[:20])


if __name__ == '__main__':
    test_add()
