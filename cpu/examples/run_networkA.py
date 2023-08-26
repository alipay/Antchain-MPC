"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""

import os
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server, start_client
from stensorflow.ml.nn.networks.NETWORKA import NETWORKA
import numpy as np
import time
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from cnn_utils import dense_to_onehot, calculate_score
from stensorflow.random.random import random_init
from sklearn.utils import shuffle

start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

epochs = 1
batch_size = 128
learning_rate = 0.01
momentum = 0.9
l2_regularzation = 1E-6


def load_data(normal=True, small=True):
    """
    load mnist data
    :return:
    """
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape((60000, 28 * 28), order='C')
    test_images = test_images.reshape((10000, 28 * 28), order='C')
    if normal:
        training_images = training_images / 255.0
        test_images = test_images / 255.0
        training_images = (training_images - 0.1307) / 0.3081
        test_images = (test_images - 0.1307) / 0.3081
    if small:
        return training_images[:6400], training_labels[:6400], test_images[:896], test_labels[:896]
    else:
        return training_images, training_labels, test_images, test_labels


def convert_datasets(train_x, train_y, test_x, test_y, epoch=1, batch_size=10):
    """
    :return: batched data
    """
    train_y = dense_to_onehot(train_y)
    train_y = train_y.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.repeat(epoch)
    train_dataset = train_dataset.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    image_batch, label_batch = iterator.get_next()
    image_batch = tf.reshape(image_batch, shape=[batch_size, 28 * 28])
    label_batch = tf.reshape(label_batch, shape=[batch_size, 10])
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
    test_image_batch, test_label_batch = test_iterator.get_next()
    test_image_batch = tf.reshape(test_image_batch, shape=[batch_size, 28 * 28])
    test_label_batch = tf.reshape(test_label_batch, shape=[batch_size, 10])
    x_train = PrivateTensor(owner='L')
    y_train = PrivateTensor(owner='R')
    x_train.load_from_tf_tensor(image_batch)
    y_train.load_from_tf_tensor(label_batch)
    x_test = PrivateTensor(owner='L')
    y_test = PrivateTensor(owner='R')
    x_test.load_from_tf_tensor(test_image_batch)
    y_test.load_from_tf_tensor(test_label_batch)
    return x_train, y_train, x_test, y_test


def keras_network_baseline(train_x, train_y, test_x, test_y):
    """
    NetWorkA using Keras
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_dim=28 * 28),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')

    ])
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    print("test result")
    # evaluate
    test_loss = model.evaluate(test_x, test_y)
    pred_y = model.predict(test_x)
    ans = []
    for pred in pred_y:
        pred = list(pred)
        ans.append(pred.index(max(pred)))
    print(ans)
    print(test_loss)
    model.save("../output/DNN_mnist_model.h5")
    pass


def stf_networkA_test(train_x, train_y, test_x, test_y, keras_weight=None):
    """
    NETWORK A using STF
    :param train_x: figure for training
    :param train_y: figure for label
    :param test_x: figure for training
    :param test_y: figure for label
    :param keras_weight:   initial weight of format of Keras
    :return:
    """
    sess = tf.compat.v1.Session(StfConfig.target)

    record_num = train_x.shape[0]
    batch_num_per_epoch = record_num // batch_size
    train_batch_num = epochs * batch_num_per_epoch
    # train_batch_num = 10
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size

    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y,
                                                        epoch=epochs, batch_size=batch_size)
    # build the model
    model = NETWORKA(feature=x_train, label=y_train)
    if keras_weight:
        # initial weight
        print("start replace")
        model.replace_weight(keras_weight)
    # compile model
    model.compile()
    print("success compile")
    print("start train model")
    start_time = time.time()
    # random_init(sess)
    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num,
                    l2_regularization=l2_regularzation,
                    sess=sess, momentum=momentum)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # model.save_model(save_file_path="../output/STF_CNN.npz", sess=sess, model_file_machine='R')
    print("start predict")
    model.predict_to_file(sess, x_test, StfConfig.predict_to_file, pred_batch_num=pred_batch_num,
                          model_file_machine='R', with_sigmoid=False)
    # print("model pram")
    # model.print(sess=sess, model_file_machine='R')


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data(normal=True, small=True)
    print(train_x, train_y, test_x, test_y)
    # train
    keras_network_baseline(train_x, train_y, test_x, test_y)
    # exit()
    # print("reading Keras model...")
    # keras_model = tf.keras.models.load_model("../output/dnn_mnist_model.h5")
    # keras_weight = keras_model.get_weights()
    # for w in keras_weight:
    #     print(w)
    # exit()
    # test
    # test_loss = keras_model.evaluate(test_x, test_y)
    # print("keras test result: " + str(test_loss))
    StfConfig.default_fixed_point = 16
    stf_networkA_test(train_x, train_y, test_x, test_y)
    calculate_score(StfConfig.predict_to_file)
