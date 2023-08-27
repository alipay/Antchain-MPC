#!/usr/bin/env python3
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2023 All Rights Reserved.
   ------------------------------------------------------
   File Name : run_vgg16_ti.py
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : Jan 14, 2023   17:05
   Description : run VGG16 over Tiny-ImageNet
"""

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client

from tensorflow.python.keras.utils import np_utils
import numpy as np

start_local_server(config_file="../conf/config.json")

import time
import tensorflow as tf
from stensorflow.ml.nn.networks.NETWORKC import NetworkC
from stensorflow.ml.nn.networks.CNN_with_SL import LocalNetworkC
from stensorflow.ml.nn.networks.VGG16 import VGG16, LocalVGG16_TI
from cnn_utils import convert_datasets, load_data_tiny_imagenet, calculate_score_tiny_imagenet

epoch = 10
batch_size = 128  # 128
learning_rate = 0.005  # 0.01
momentum = 0.8
l2_regularzation = None  # 0.01
samples_num = 100000



def keras_baseline(train_x, train_y, test_x, test_y, train=True, save_model_path="../output/vgg16_ti_keras.h5"):
    """
    network C using Keras
    :return:
    """
    use_bias = False
    pooling_layer = tf.keras.layers.AveragePooling2D
    # pooling_layer = tf.keras.layers.MaxPooling2D

    if train:
        model = tf.keras.models.Sequential([
            # 1
            # tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3),
            #                        use_bias=use_bias),

            # First Layer
            tf.keras.layers.Conv2D(20, (5, 5), activation='relu', input_shape=(64, 64, 3), use_bias=use_bias),
            tf.keras.layers.MaxPooling2D(2, 2) if pooling_layer == 'max' else tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Conv2D(50, (5, 5), activation='relu', use_bias=False),
            tf.keras.layers.MaxPooling2D(2, 2) if pooling_layer == 'max' else tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Flatten(),
            # Third layer
            tf.keras.layers.Dense(500, activation='relu'),
            # Final Layer
            tf.keras.layers.Dense(200, name="Dense"),
            tf.keras.layers.Activation('softmax')
        ])

        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        # opt = tf.keras.optimizers.RMSprop()
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        print("start train model")
        start_time = time.time()
        model.fit(train_x, train_y, epochs=int(epoch), batch_size=batch_size)
        end_time = time.time()
        print("train time=", end_time - start_time)
        # print(model.get_weights())
        # test result
        print("test result")
        # evaluate
        test_loss = model.evaluate(test_x, test_y)
        print("test result: " + str(test_loss))
        model.save(save_model_path)
    else:
        print("train 1 epoch using model load")
        keras_model = tf.keras.models.load_model("../output/networkC_ti_keras.h5")
        train_x = train_x[:batch_size]
        train_y = train_y[:batch_size]
        Epoch = 1
        for _ in range(Epoch):
            keras_model.fit(train_x, train_y, epochs=1, batch_size=batch_size)
        test_loss = keras_model.evaluate(test_x, test_y)
        print("keras test result: " + str(test_loss))
        keras_model.save("../output/networkC_ti_keras.h5")


def stf_cnn_test(train_x, train_y, test_x, test_y, epochs, load_model=None, save_model_path="../output/complex_CNN.npz",
                 pooling='avg', l2_regularzation=None):
    """
    NETWORK D using STF
    :param train_x: figure for training
    :param train_y: figure for label
    :param test_x: figure for training
    :param test_y: figure for label
    :param load_model:   initial weight of format of Keras
    :return:
    """
    sess = tf.compat.v1.Session(StfConfig.target)
    record_num = train_x.shape[0]
    batch_num_per_epoch = record_num // batch_size
    train_batch_num = int(epochs * batch_num_per_epoch)
    # train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    # convert the data
    # print("l113, train_x=", train_x)
    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y, size=(64, 64, 3),
                                                        epoch=epochs, batch_size=batch_size, classes_num=200)

    print("l116 xtrain=", x_train.shape)
    print("l117 y_train=", y_train.shape)
    # build model
    # model = VGG16(feature=x_train, label=y_train, pooling=pooling)
    # model = LocalVGG16_TI(feature=x_train, label=y_train, pooling=pooling)
    # model = NetworkC(feature=x_train, label=y_train, pooling=pooling, input_shape=[64, 64, 3])
    model = LocalNetworkC(feature=x_train, label=y_train, input_shape=[64, 64, 3], output_dim=200)

    # pred = model.predict(x_test)
    # print("pred=", pred)
    if load_model is not None:
        print("start replace")
        model.replace_weight(load_model)
    # compile model
    model.compile()
    print("success compile")
    print("start train model")
    start_time = time.time()
    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularzation,
                    sess=sess, momentum=momentum)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # random.random_init(sess)
    print("start predict")
    model.predict_to_file(sess, x_test, StfConfig.predict_to_file, pred_batch_num=pred_batch_num,
                          model_file_machine='R', out_prob=False)
    if save_model_path:
        model.save_model(save_file_path=save_model_path, sess=sess, model_file_machine='R')


if __name__ == "__main__":
    StfConfig.default_fixed_point = 14  # 12  14
    StfConfig.softmax_iter_num = 32
    train_x, train_y, test_x, test_y = load_data_tiny_imagenet(normal=True, small=False)

    print("x_train.shape=", train_x.shape)
    print("y_train.shape=", train_y.shape)

    keras_weight = None

    # learning_rate = 1.0/(1<<16)
    StfConfig.truncation_functionality = True
    StfConfig.softmax_functionality = True

    # keras_baseline(train_x, train_y, test_x, test_y, train=True)


    stf_cnn_test(train_x, train_y, test_x, test_y, epochs=epoch, load_model=keras_weight,
                 save_model_path="../output/complex_CNN.npz")

    calculate_score_tiny_imagenet(StfConfig.predict_to_file)
