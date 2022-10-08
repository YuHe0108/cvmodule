import cv2
import time
import math
import numba
from numba import cuda


# GPU function
@cuda.jit
def process_gpu(img, rows, cols, channels):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if tx < rows and ty < cols:
        for c in range(channels):
            color = img[tx, ty][c] * 2.0 + 30
            if color > 255:
                img[tx, ty][c] = 255
            elif color < 0:
                img[tx, ty][c] = 0
            else:
                img[tx, ty][c] = color
    return


if __name__ == '__main__':
    filename = r'C:\Users\yuhe\Desktop\draw\NegativeSamples\0b15da1c-eee8-4574-9cf8-62e4e8509f1b.jpg'
    for _ in range(3):
        t = time.time()
        img = cv2.imread(filename)
        rows, cols, channels = img.shape

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(math.ceil(rows / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(cols / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        start_gpu = time.time()
        dImg = cuda.to_device(img)
        cuda.synchronize()
        process_gpu[blocks_per_grid, threads_per_block](dImg, rows, cols, channels)
        res = dImg.copy_to_host()
        print(time.time() - t)
