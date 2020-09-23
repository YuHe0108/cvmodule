"""实现训练分类模型的抽象类"""
import tensorflow as tf
import abc


class ClassifierModule(abc.ABC):
    def __init__(self, *args, **kwargs):
        tf.keras.backend.set_learning_phase(True)
        pass

    @abc.abstractclassmethod
    def configure_optimizer(cls, optimizer):
        """子类必须要实现的类"""
        return

    def load_weights(self):
        pass

    @tf.function
    def train_step(self, train_data, train_label):
        pass

    @tf.function
    def train_step(self, train_data, train_label):
        pass

    def train(self):
        pass

    def test(self):
        pass



