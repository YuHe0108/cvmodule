"""
SE-Net: 通道上的注意力
    (1) 显式地建立模型，定义通道间关系
    (2) 实现"特征重标定", 即对于不同channel-wise，加强有用信息并压缩无用信息
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_package.Module import se_block


def squeeze_module():
    pass
