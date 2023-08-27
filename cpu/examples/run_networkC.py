#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_networkC
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/9/2 上午11:08
   Description : description what the main function of this file
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=float, default=1)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pooling', type=str, default='avg')
parser.add_argument('--predict_flag', type=int, default=1)
parser.add_argument('--truncation_type', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--config_file', type=str, default="../conf/config.json")
args = parser.parse_args()
print("network C")
print(args)

import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client
start_local_server(config_file=args.config_file)
# start_client(config_file="../conf/config.json", job_name="workerR")
import time
import tensorflow as tf
from stensorflow.ml.nn.networks.NETWORKC import NetworkC
from stensorflow.ml.nn.networks.CNN_with_SL import LocalNetworkC
from cnn_utils import convert_datasets, load_data_mnist, calculate_score_mnist
from sklearn.metrics import accuracy_score


epochs = 1
batch_size = 128
learning_rate = 0.01
momentum = 0.9
l2_regularzation =1E-6
pooling = 'avg'
def cnn_baseline(train_x, train_y, test_x, test_y, epochs, pooling='max'):
    """
    network C using Keras
    :return:
    """

    model = tf.keras.models.Sequential([
        # First Layer
        tf.keras.layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1), use_bias=False),
        tf.keras.layers.MaxPooling2D(2,2) if pooling=='max' else tf.keras.layers.AvgPool2D(2, 2),
        tf.keras.layers.Conv2D(50, (5, 5), activation='relu', use_bias=False),
        tf.keras.layers.MaxPooling2D(2,2) if pooling=='max' else tf.keras.layers.AvgPool2D(2, 2),
        tf.keras.layers.Flatten(),
        # Third layer
        tf.keras.layers.Dense(500, activation='relu'),
        # Final Layer
        tf.keras.layers.Dense(10, name="Dense"),
        tf.keras.layers.Activation('softmax')
    ])
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("start train model")
    start_time = time.time()
    model.fit(train_x, train_y, epochs=int(epochs), batch_size=batch_size)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # evaluate
    # test_loss = model.evaluate(test_x, test_y)
    pred_y = model.predict(test_x)
    ans = []
    for pred in pred_y:
        pred = list(pred)
        ans.append(pred.index(max(pred)))

    # evaluate
    test_acc = accuracy_score(test_y, ans)
    # print(ans)
    print("test_acc=", test_acc)
    model.save("../output/complex_mnist_model.h5")


def stf_cnn_test(train_x, train_y, test_x, test_y, epochs, keras_weight=None, pooling='max', learning_rate=0.01, predict_flag=True):
    """
    NETWORK B using STF
    :param train_x: figure for training
    :param train_y: figure for label
    :param test_x: figure for training
    :param test_y: figure for label
    :param keras_weight:   initial weight of format of Keras
    :return:
    """
    sess = tf.compat.v1.Session(StfConfig.target)

    # epochs = 5
    # batch_size = 256
    # learning_rate = 0.01  # default learning rate in keras SGD
    record_num = train_x.shape[0]  # number of traininig samples
    batch_num_per_epoch = record_num / batch_size
    train_batch_num = int(epochs * batch_num_per_epoch)
    # train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    # convert data
    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y,
                                                        epoch=epochs, batch_size=batch_size)
    # build model
    model = NetworkC(feature=x_train, label=y_train, pooling=pooling)
    if keras_weight:
        # load weights
        print("start replace")
        model.replace_weight(keras_weight)
    # compile model
    model.compile()
    print("success compile")
    print("start train model")

    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num,
                    l2_regularization=l2_regularzation, sess=sess, momentum=momentum)

    if predict_flag:
        print("start predict")
        model.predict_to_file(sess, x_test, StfConfig.predict_to_file, pred_batch_num=pred_batch_num,
                            model_file_machine='R', out_prob=False)
        model.save_model(save_file_path="../output/complex_CNN.npz", sess=sess, model_file_machine='R')


if __name__ == "__main__":

    np.random.seed(0)
    epochs = args.epochs
    pretrain = args.pretrain
    pooling = args.pooling
    predict_flag = (args.predict_flag==1)
    train_x, train_y, test_x, test_y = load_data_mnist(normal=True, small=False)
    # print(train_x, train_y, test_x, test_y)

    if pretrain == 1:
        cnn_baseline(train_x, train_y, test_x, test_y, max(epochs-1, 1), pooling=pooling)

        print("reading Keras model...")
        keras_model = tf.keras.models.load_model("../output/complex_mnist_model.h5")
        keras_weight = keras_model.get_weights()

        stf_cnn_test(train_x, train_y, test_x, test_y, 1, keras_weight, pooling=pooling, learning_rate=0.001, predict_flag=predict_flag)
    else:
        stf_cnn_test(train_x, train_y, test_x, test_y, epochs, None, pooling=pooling, learning_rate=learning_rate, predict_flag=predict_flag)

    if predict_flag:
        calculate_score_mnist(StfConfig.predict_to_file)