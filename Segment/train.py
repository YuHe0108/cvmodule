import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import pathlib
import time
import tqdm
import os

from tf_package import utils
import dataset
import model

# 1. 获取图像分割模型
seg_model = model.deeplab_v3((256, 256, 3), classes=3, activation='softmax', pre_train=False)
seg_model.summary()

# 2. 设置优化器和保存路径
train_steps = tf.Variable(0, tf.int32)
model_optimizer = tf.keras.optimizers.Adam(0.0002)
model_checkpoint_dir = os.path.join('', 'checkpoint')
model_checkpoint = tf.train.Checkpoint(train_steps=train_steps, seg_model=seg_model, model_optimizer=model_optimizer)
model_manger = tf.train.CheckpointManager(model_checkpoint, directory=model_checkpoint_dir, max_to_keep=1)


def loss_fun(pred, target):
    return tf.losses.sparse_categorical_crossentropy(y_pred=pred, y_true=target)


def create_mask(pred_mask):
    """返回预测的图像"""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]  # [b, h, w, 1]
    return pred_mask[0]


def load_model():
    if model_manger.latest_checkpoint:
        model_checkpoint.restore(model_manger.latest_checkpoint)
        print('加载模型: {}'.format(model_manger.latest_checkpoint))
        print('模型之前已经经过{}次训练，训练继续！'.format(model_checkpoint.train_steps.numpy()))
    else:
        print('重新训练模型！')


# 4.测试数据集、数据可视化
# 在训练过程中, 每隔100batch，测试模型的分割性能
train_image_save_path = 'train_pred_images'
# 在模型训练完成之后，测试模型的分割性能
if not os.path.exists(train_image_save_path):
    os.mkdir(train_image_save_path)


# 5. 设置训练step
@tf.function
def test_step(test_image):
    tf.keras.backend.set_learning_phase(False)
    test_pred = seg_model(test_image)
    return test_pred


@tf.function
def train_step(image, target):
    tf.keras.backend.set_learning_phase(True)
    with tf.GradientTape() as tape:
        pred = seg_model(image)
        loss = loss_fun(pred, target)
    gradient = tape.gradient(loss, seg_model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradient, seg_model.trainable_variables))
    return tf.reduce_mean(loss)


def test(test_data):
    test_losses = []
    for test_image, test_target in test_data:
        test_pred = test_step(test_image)
        test_loss = tf.reduce_mean(loss_fun(test_pred, test_target))
        test_losses.append(test_loss)
    return test_pred, tf.reduce_mean(test_losses)


def train(train_data, test_data, epochs=100):
    load_model()
    pd_record = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss', 'Time'])
    # data_name, original_test_image, norm_test_image = data.get_test_data(test_data_path=test_image_path,
    #                                                                      image_shape=input_shape,
    #                                                                      image_nums=-1)

    start_time = time.time()
    for epoch in range(epochs):
        for batch_index, (image, mask) in enumerate(tqdm.tqdm(train_data)):
            model_checkpoint.train_steps.assign_add(1)
            iteration = model_checkpoint.train_steps.numpy()

            train_loss = train_step(image, mask)

            if iteration % 100 == 0 or iteration == 1:
                print('Epoch: {}, Iteration: {}, Loss: {:.2f}, Time: {:.2f} s'.format(
                    epoch + 1, iteration, train_loss, time.time() - start_time))

                # 测试模型性能
                test_image, test_target = test_data.next()
                test_pred = test_step(test_image)
                test_pred = tf.argmax(test_pred, -1)

                test_grid = utils.make_gird(test_image, n_rows=4, denorm=True, denorm_rang='-11')
                utils.save_samples(test_grid, 'test\\image', epoch, iteration, i=batch_index, denorm=False)

                test_target_grid = utils.make_gird(test_target + 1, n_rows=4)
                utils.save_samples(test_target_grid, 'test\\target', epoch, iteration, i=batch_index, denorm=False)

                pred_grid = utils.make_gird(test_pred + 1, n_rows=4)
                utils.save_samples(pred_grid, 'test\\predict', epoch, iteration, i=batch_index, denorm=False)

                # 保存数据到csv文件中
                pd_record = pd_record.append({'Epoch': epoch + 1, 'Iteration': iteration, 'Loss': train_loss.numpy(),
                                              'Time': time.time() - start_time}, ignore_index=True)
                pd_record.to_csv('{}_record.csv'.format('muti_seg_model'), index=True, header=True)

        model_manger.save(checkpoint_number=epoch + 1)
    return 0


def predict(image_dir, batch_size=4, epoch=1):
    # 从指定路径下读取图像，并预测, 用于预测分割图像
    load_model()
    images = []
    for path in pathlib.Path(image_dir).iterdir():
        image = cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB)
        image = np.float32(image) / 127.5 - 1
        image = cv.resize(image, (256, 256))
        images.append(np.expand_dims(image, 0))

    batch_data = np.concatenate(images[:batch_size], 0)
    pred_image = test_step(batch_data)
    pred_image = tf.argmax(pred_image, -1)
    grid_image = utils.make_gird(pred_image, n_rows=2)
    utils.save_samples(grid_image, 'predict_images', epoch, color='rgb', denorm=False)
    return


if __name__ == '__main__':
    TRAIN = False
    data_root_dir = r'J:\DATA\OxfordCat'
    if TRAIN:
        train_data_ = dataset.get_tfrecord_data(img_shape=(256, 256, 3), re_size=256, batch_size=16,
                                                tf_record_path=os.path.join(data_root_dir, 'train_seg_256_7390'),
                                                repeat=1, buffer_size=300, norm=True, is_train_data=False)
        test_data_ = dataset.get_tfrecord_data(img_shape=(256, 256, 3), re_size=256, batch_size=16,
                                               tf_record_path=os.path.join(data_root_dir, 'test_seg_256_7390'),
                                               repeat=-1, buffer_size=100, norm=True, is_train_data=False)
        train(train_data_, iter(test_data_), 100)

    else:
        eval_image_path = r'inval_images'
        predict(eval_image_path)
