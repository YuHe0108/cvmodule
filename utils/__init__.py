import numpy as np
import cv2

array = np.zeros(shape=(256, 256), dtype=np.uint8)
array[0][0] = 10
array[60:150][60:150] = 10000

res = cv2.applyColorMap(array, cv2.COLORMAP_HOT)
cv2.imshow("res", res)
cv2.waitKey(0)