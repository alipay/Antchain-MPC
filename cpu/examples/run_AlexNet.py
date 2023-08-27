#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : run_Alexnet.py
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
print("AlexNet")
print(args)

from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client


from tensorflow.python.keras.utils import np_utils
import numpy as np
start_local_server(config_file=args.config_file)

# start_client(config_file=args.config_file, job_name="workerR")
import time
import tensorflow as tf
from stensorflow.ml.nn.networks.AlexNet import AlexNet
from cnn_utils import convert_datasets, load_data_mnist, calculate_score_mnist, load_data_cifar10, calculate_score_cifar10
from sklearn.metrics import accuracy_score


epoch = 10 # 5
batch_size = 32 # 128
learning_rate = 0.01
momentum = 0.8
l2_regularzation = 0.01


def cnn_baseline(train_x, train_y, test_x, test_y, epochs, pooling='max'):
    """
    network C using Keras
    :return:
    """
    use_bias = False
    model = tf.keras.models.Sequential([

        # tf.keras.layers.ZeroPadding2D((9, 9), input_shape=(32, 32, 3)),
        # tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu', input_shape=(32, 32, 3), use_bias=use_bias),
        tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2)),

        tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same', use_bias=use_bias),
        tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2)),

        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same', use_bias=use_bias),
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same', use_bias=use_bias),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', use_bias=use_bias),


        tf.keras.layers.Flatten(),
        # Third layer
        tf.keras.layers.Dense(256, activation='relu'),
        # Final Layer
        tf.keras.layers.Dense(256, activation='relu'),
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


def stf_cnn_test(train_x, train_y, test_x, test_y, epochs, keras_weight=None, learning_rate=0.01, batch_size=32, momentum=0.0):
    """
    AlexNet using STF
    :param train_x: figure for training
    :param train_y: figure for label
    :param test_x: figure for training
    :param test_y: figure for label
    :param keras_weight:   initial weight of format of Keras
    :return:
    """
    sess = tf.compat.v1.Session(StfConfig.target)
    record_num = train_x.shape[0]
    batch_num_per_epoch = record_num / batch_size
    train_batch_num = int(epochs * batch_num_per_epoch)
    # train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    # convert data
    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y, size=(32, 32, 3),
                                                        epoch=epochs, batch_size=batch_size, classes_num=10)

    # build model
    model = AlexNet(feature=x_train, label=y_train)

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
    truncation_type = args.truncation_type
    batch_size = args.batch_size
    momentum = args.momentum
    
    
    StfConfig.default_fixed_point = 24
    if truncation_type==1:
        StfConfig.truncation_functionality = True
        
    train_x, train_y, test_x, test_y = load_data_cifar10(normal=False, small=False)

    if pretrain == 1:
        cnn_baseline(train_x, train_y, test_x, test_y, max(epochs - 1, 1), pooling=pooling)

        print("reading Keras model...")
        keras_model = tf.keras.models.load_model("../output/complex_mnist_model.h5")
        keras_weight = keras_model.get_weights()

        stf_cnn_test(train_x, train_y, test_x, test_y, 1, keras_weight, learning_rate=0.0025, batch_size=batch_size, momentum=momentum)
    else:
        stf_cnn_test(train_x, train_y, test_x, test_y, epochs, None, learning_rate=learning_rate, batch_size=batch_size, momentum=momentum)

    if predict_flag:
        calculate_score_cifar10(StfConfig.predict_to_file)