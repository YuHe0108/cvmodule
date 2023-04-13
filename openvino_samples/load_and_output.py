import sys
import time
import cv2
import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from openvino.runtime import Core

DEVICE = "CPU"
MODEL_FILE = "/home/yuhe/Projects/Models/public/midasnet/openvino_midas_v21_small_256.xml"
model_xml_path = Path(MODEL_FILE)
model_xml_path = Path(MODEL_FILE)


def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.colormaps.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


ie = Core()
ie.set_property({'CACHE_DIR': './cache'})
model = ie.read_model(model_xml_path)
compiled_model = ie.compile_model(model=model, device_name=DEVICE)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]

IMAGE_FILE = "/home/yuhe/Images/plane.png"
image = cv2.imread(IMAGE_FILE)

# Resize to input shape for network.
resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))

# Reshape the image to network input shape NCHW.
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

result = compiled_model([input_image])[output_key]
# Convert the network result of disparity map to an image that shows
# distance as colors.
result_image = convert_result_to_image(result=result)

# in (width, height), [::-1] reverses the (height, width) shape to match this.
result_image = cv2.resize(result_image, image.shape[:2][::-1])

fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].imshow(to_rgb(image))
ax[1].imshow(result_image)
plt.show()
