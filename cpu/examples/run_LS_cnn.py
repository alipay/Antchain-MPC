"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""
import os
from stensorflow.engine.start_server import start_local_server
start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))
from stensorflow.ml.nn.networks.CNN_with_SL import *
import time
from cnn_utils import *


def average_cnn_baseline(train_x, train_y, test_x, test_y, train = True):
    """
    baseline: CNN using keras on dataset minist.
    :return:
    """
    if train:
        model = tf.keras.models.Sequential([
            # First Layer
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1), use_bias=False),
            tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', use_bias=False),
            tf.keras.layers.AvgPool2D(2, 2),
            tf.keras.layers.Flatten(),
            # Third layer
            tf.keras.layers.Dense(100, activation='relu'),
            # Final Layer
            tf.keras.layers.Dense(10, name="Dense"),
            tf.keras.layers.Activation('softmax')
        ])
        sgd = tf.keras.optimizers.SGD(lr=0.01)
        model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(train_x, train_y, epochs=10, batch_size=128)
        # test result
        print("test result")
        # evaluate
        test_loss = model.evaluate(test_x, test_y)
        print("test result: " + str(test_loss))
        model.save("../output/LS_mnist_model.h5")
    else:
        print("train 1 epoch in given initial weights")
        keras_model = tf.keras.models.load_model("../output/LS_mnist_model.h5")
        train_x = train_x[:128]
        train_y = train_y[:128]
        keras_model.fit(train_x, train_y, epochs=1, batch_size=128)
        test_loss = keras_model.evaluate(test_x, test_y)
        print("keras test result: " + str(test_loss))
        keras_model.save("../output/epoch.h5")


def stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None):
    sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    Epochs = 10
    batch_size = 128
    learning_rate = 0.01 # default learning rate in keras SGD
    record_num = train_x.shape[0]  # number of training samples
    batch_num_per_epoch = record_num // batch_size
    # train_batch_num = Epochs * batch_num_per_epoch
    train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    predict_file_name = "../output/LS_mnist_predict.txt"
    # load data
    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y,
                                                        epoch=Epochs, batch_size=batch_size)
    # load model
    model = LocalCNN(feature=x_train, label=y_train, loss="CrossEntropyLossWithSoftmax")
    # model = Local_Complex_CNN(feature=x_train, label=y_train, loss="CrossEntropyLossWithSoftmax")
    if keras_weight:
        # load keras weight
        print("start replace")
        model.replace_weight(keras_weight)
    model.compile()
    print("start train model")
    start_time = time.time()
    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=0, sess=sess)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # random.random_init(sess)
    print("start predict")
    model.predict_to_file(sess, x_test, predict_file_name, pred_batch_num=pred_batch_num,
                          model_file_machine='R', record_num_ceil_mod_batch_size=batch_size, with_sigmoid=False)

    # model.save_model(save_file_path="../output/LS_CNN.npz", sess=sess, model_file_machine='R')


if __name__ == "__main__":
    # compare_weight(keras_model_path="../output/epoch.h5",
    #                stf_model_path="../output/LS_CNN.npz")
    # exit()
    train_x, train_y, test_x, test_y = load_data_mnist(normal=True, small=True)

    # baseline
    # average_cnn_baseline(train_x, train_y, test_x, test_y, train=False)
    # exit()
    # print("reading model from Keras...")
    # keras_model = tf.keras.models.load_model("../output/LS_mnist_model.h5")
    # keras_weight = keras_model.get_weights()
    # exit()
    # test
    # test_loss = keras_model.evaluate(test_x, test_y)
    # print("keras test result: " + str(test_loss))
    stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None)
    calculate_score_mnist("../output/LS_mnist_predict.txt")
    # exit()
    # compare_forward(keras_model_path="../output/LS_mnist_model.h5",
    #                 stf_predict_path="../output/LS_mnist_predict.txt",
    #                 test_x=test_x)
    # exit()

