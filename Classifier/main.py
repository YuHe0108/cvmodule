# 使用迁移学习对 Brain 图像进行分类
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import tqdm
import data
import os

DATA_NAME = 'BRAIN'
BATCH_SIZE = 32
RESIZE = 256
INPUT_CHANNEL = 3
TRAIN_TFRECORD_PATH = r'J:\DATA\Brain\TF_Records\Classifier_Train_Test\train'
TEST_TFRECORD_PATH = r'J:\DATA\Brain\TF_Records\Classifier_Train_Test\test'
CLASSES = 4
LR = 0.0002
MODEL_SAVE_PATH = 'classifier_ckpt'
EPOCHS = 20
INITIAL_EPOCHS = 0
TOTAL_SAMPLES = 1785 + 3600 + 4454 + 4705
TEST_SAMPLES = 199 + 400 + 495 + 523

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

tf.keras.backend.set_learning_phase(True)


# 2、迁移学习模型
def transfer_model(input_shape, classes):
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      pooling='avg',
                                                      weights='imagenet',
                                                      input_shape=input_shape)
    print('Base-模型层数一共有: ', len(base_model.layers))

    x = keras.layers.Dense(512)(base_model.layers[-1].output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(classes, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    print('总共模型的层数一共有: ', len(model.layers))
    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:  # 倒数十层之后的模型不训练
        layer.trainable = True
    return model


classifier_model = transfer_model((RESIZE, RESIZE, INPUT_CHANNEL), CLASSES)
classifier_model.summary()
optim = keras.optimizers.Adam(LR)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred = classifier_model(images, training=True)
        loss = keras.losses.categorical_crossentropy(y_true=labels, y_pred=pred)
        accu = keras.metrics.categorical_accuracy(y_true=labels, y_pred=pred)

    grads = tape.gradient(loss, classifier_model.trainable_variables)
    optim.apply_gradients(zip(grads, classifier_model.trainable_variables))
    return loss, accu


def test_step(images, labels):
    pred = classifier_model(images, training=True)
    loss = keras.losses.categorical_crossentropy(y_true=labels, y_pred=pred)
    accu = keras.metrics.categorical_accuracy(y_true=labels, y_pred=pred)
    return loss, accu


def lr_fn(epoch):
    if epoch <= 10:
        lr = LR
    elif 10 < epoch <= 16:
        lr = 0.0001
    else:
        lr = 0.00005
    print('Learning rate: ', lr)
    return lr


def load_model_weight(weight_path):
    if os.path.isfile(weight_path):
        classifier_model.load_weights(weight_path)
        print('加载成功')
    else:
        print('重新训练')
    return


def train(train_data, test_data):
    data_frame = pd.DataFrame(columns=['Epoch', 'train_loss', 'train_accu', 'test_loss', 'test_accu'])
    for epoch in range(EPOCHS):

        optim.learning_rate = lr_fn(epoch)
        print('学习率: ', optim.learning_rate.numpy())

        train_loss, train_accu = [], []
        test_loss, test_accu = [], []

        for index, (train_image, train_label) in tqdm.tqdm(enumerate(train_data), total=TOTAL_SAMPLES // BATCH_SIZE):
            if len(train_label.shape) == 1:
                train_label = keras.utils.to_categorical(train_label, CLASSES)
            train_step_loss, train_step_accu = train_step(train_image, train_label)
            train_loss.append(train_step_loss)
            train_accu.append(train_step_accu)

        for test_image, test_label in tqdm.tqdm(test_data, total=TEST_SAMPLES // BATCH_SIZE):
            if len(test_label.shape) == 1:
                test_label = keras.utils.to_categorical(test_label, CLASSES)
            test_step_loss, test_step_accu = test_step(test_image, test_label)

            test_loss.append(test_step_loss)
            test_accu.append(test_step_accu)

        mean_test_loss = np.mean(test_loss)
        mean_test_accu = np.mean(test_accu)
        mean_train_loss = np.mean(train_loss)
        mean_train_accu = np.mean(train_accu)

        print('Epoch: {}, train_accu: {:.2f}, test_accu: {:.2f}'.format(epoch, mean_train_accu, mean_test_accu))
        classifier_model.save_weights(os.path.join(MODEL_SAVE_PATH, '{:03d}.h5'.format(epoch)))
        data_frame = data_frame.append({
            'epoch': epoch + 1, 'train_loss': mean_train_loss, 'train_accu': mean_train_accu,
            'test_loss': mean_test_loss, 'test_accu': mean_test_accu,
        }, ignore_index=True)
        data_frame.to_csv('record.csv', index=True, header=True)

    return


if __name__ == '__main__':
    # 1、获得数据集
    train_data_ = data.get_data(batch_size=BATCH_SIZE, re_size=RESIZE, data_name=DATA_NAME, repeat=1,
                                tfrecord_path=TRAIN_TFRECORD_PATH, ori_size=(RESIZE, RESIZE, INPUT_CHANNEL),
                                augment=True)
    test_data_ = data.get_data(batch_size=BATCH_SIZE, re_size=RESIZE, data_name=DATA_NAME, repeat=1,
                               tfrecord_path=TEST_TFRECORD_PATH, ori_size=(RESIZE, RESIZE, INPUT_CHANNEL))
    # train(train_data_, test_data_)
    for image, label in train_data_.take(1):
        plt.imshow(np.uint8(np.squeeze(image[0])), cmap='gray')
        plt.show()
