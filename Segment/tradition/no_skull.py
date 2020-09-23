from tf_package import utils
import cv2 as cv
import pathlib
import os


def no_skull_method(path_dir, save_dir, resize_shape=(256, 256)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 30))
    for path in pathlib.Path(path_dir).iterdir():
        image = cv.imread(str(path))
        image = cv.resize(image, resize_shape)
        erode_result = cv.erode(image, kernel)
        erode_result_gray = cv.cvtColor(erode_result, cv.COLOR_BGR2GRAY)
        threshold, mask_no_skull = cv.threshold(erode_result_gray, 1, 255, cv.THRESH_BINARY)
        result = cv.bitwise_and(image, image, mask=mask_no_skull)
        cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(path.stem)), result)
    return


if __name__ == '__main__':
    no_skull_method(r'invalid\invalid_data\image', 'no_skull_image')
