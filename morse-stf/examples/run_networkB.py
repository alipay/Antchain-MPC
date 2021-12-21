"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""
import os
from stensorflow.engine.start_server import start_local_server, start_client
start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

import time
import tensorflow as tf
from stensorflow.ml.nn.networks.NETWORKB import NetworkB
from stensorflow.global_var import StfConfig
from cnn_utils import convert_datasets, load_data, calculate_score



epochs = 1
batch_size = 128
learning_rate = 0.01
momentum = 0.9
l2_regularzation =1E-6
pooling = 'avg'

def cnn_baseline(train_x, train_y, test_x, test_y, train=True, pooling='max'):
    """
    network B using Keras
    :return:
    """
    if train:
        sq = [
            # First Layer
            tf.keras.layers.Conv2D(16, (5, 5), input_shape=(28, 28, 1), use_bias=False),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2) if pooling=='max' else tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Conv2D(16, (5, 5), use_bias=False),
            #tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2) if pooling=='max' else tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Flatten(),
            # Third layer
            tf.keras.layers.Dense(100),
            #tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation='relu'),
            # Final Layer
            tf.keras.layers.Dense(10, name="Dense"),
            tf.keras.layers.Activation('softmax')
        ]
        model = tf.keras.models.Sequential(sq)
        sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
        model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        print("start train model")
        start_time = time.time()
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
        end_time = time.time()
        print("train time=", end_time - start_time)
        # print(model.get_weights())
        # test result
        print("test result")
        # evaluate
        test_loss = model.evaluate(test_x, test_y)
        print("test result: " + str(test_loss))
        # model.save("../output/complex_mnist_model.h5")
    else:
        print("train 1 epoch using model load")
        keras_model = tf.keras.models.load_model("../output/complex_mnist_model.h5")
        train_x = train_x[:batch_size]
        train_y = train_y[:batch_size]
        keras_model.fit(train_x, train_y, epochs=1, batch_size=batch_size)
        test_loss = keras_model.evaluate(test_x, test_y)
        print("keras test result: " + str(test_loss))
        keras_model.save("../output/complex_epoch.h5")
#
# def max_cnn_baseline(train_x, train_y, test_x, test_y, train=True):
#     """
#     network B using Keras
#     :return:
#     """
#     if train:
#         model = tf.keras.models.Sequential([
#             # First Layer
#             tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1), use_bias=False),
#             tf.keras.layers.MaxPool2D(2, 2),
#             tf.keras.layers.Conv2D(16, (5, 5), activation='relu', use_bias=False),
#             tf.keras.layers.MaxPool2D(2, 2),
#             tf.keras.layers.Flatten(),
#             # Third layer
#             tf.keras.layers.Dense(100, activation='relu'),
#             # Final Layer
#             tf.keras.layers.Dense(10, name="Dense"),
#             tf.keras.layers.Activation('softmax')
#         ])
#         sgd = tf.keras.optimizers.SGD(lr=0.01)
#         model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         model.summary()
#         print("start train model")
#         start_time = time.time()
#         model.fit(train_x, train_y, epochs=10, batch_size=128)
#         end_time = time.time()
#         print("train time=", end_time - start_time)
#         # print(model.get_weights())
#         # test result
#         print("test result")
#         # evaluate
#         test_loss = model.evaluate(test_x, test_y)
#         print("test result: " + str(test_loss))
#         # model.save("../output/complex_mnist_model.h5")
#     else:
#         print("train 1 epoch using model load")
#         keras_model = tf.keras.models.load_model("../output/complex_mnist_model.h5")
#         train_x = train_x[:128]
#         train_y = train_y[:128]
#         keras_model.fit(train_x, train_y, epochs=1, batch_size=128)
#         test_loss = keras_model.evaluate(test_x, test_y)
#         print("keras test result: " + str(test_loss))
#         keras_model.save("../output/complex_epoch.h5")


def stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None, pooling='max'):
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
    batch_num_per_epoch = record_num // batch_size
    train_batch_num = epochs * batch_num_per_epoch
    # train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    # convert data
    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y,
                                                        epoch=epochs, batch_size=batch_size)
    # build model
    model = NetworkB(feature=x_train, label=y_train, pooling=pooling)
    if keras_weight:
        # load weights
        print("start replace")
        model.replace_weight(keras_weight)
    # compile model
    model.compile()
    print("success compile")
    print("start train model")
    start_time = time.time()
    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num,
                    l2_regularization=l2_regularzation, sess=sess, momentum=momentum)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # random.random_init(sess)
    print("start predict")
    model.predict_to_file(sess, x_test, StfConfig.predict_to_file, pred_batch_num=pred_batch_num,
                          model_file_machine='R', with_sigmoid=False)
    model.save_model(save_file_path="../output/complex_CNN.npz", sess=sess, model_file_machine='R')


if __name__ == "__main__":
    # budild_mnist("../dataset/")
    # LeNet_network()
    # exit()
    # load data
    StfConfig.default_fixed_point = 16
    StfConfig.softmax_iter_num = 40
    train_x, train_y, test_x, test_y = load_data(normal=True, small=True)
    cnn_baseline(train_x, train_y, test_x, test_y, train=True, pooling=pooling)
    # exit()
    # print("loding model...")
    # keras_model = tf.keras.models.load_model("../output/complex_mnist_model.h5")
    # keras_weight = keras_model.get_weights()
    # test_loss = keras_model.evaluate(test_x, test_y)
    # print("keras test result: " + str(test_loss))
    # exit()
    stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None, pooling=pooling)
    print("predict_to_file=", StfConfig.predict_to_file)
    calculate_score(StfConfig.predict_to_file)
    # compare_forward(keras_model_path="../output/complex_mnist_model.h5",
    #                 stf_predict_path="../output/complex_mnist_predict.txt",
    #                 test_x=test_x)
    # compare_weight(keras_model_path="../output/complex_epoch.h5",
    #                stf_model_path="../output/complex_CNN.npz")

