import os
import re
import yaml
import glob
import math
import time
import random
import shutil
import signal
import urllib
import logging
import platform
import cv2 as cv
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path
import pkg_resources as pkg
from zipfile import ZipFile
from itertools import repeat
from subprocess import check_output
from multiprocessing.pool import ThreadPool

import torch
import torchvision

# from downloads import gsutil_getsize
# from metrics import box_iou, fitness

# Settings
FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5 root directory
DATASETS_DIR = os.path.join(ROOT.parent, 'datasets')  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)  # OpenMP max threads (PyTorch and SciPy)


def is_writeable(dir_path, test=False):
    """
    测试当前路径如否可写
    Return True if directory has written permissions, test opening a file with write permissions if test=True
    """
    if test:  # method 1
        file = Path(dir_path) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging('yolov5')  # define globally (used in train.py, val.py, detect.py, etc.)


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


# CONFIG_DIR = user_config_dir()  # Ultralytics settings dir




if __name__ == '__main__':
    env = os.getenv('YOLOV5_CONFIG_DIR')
    print(env)
