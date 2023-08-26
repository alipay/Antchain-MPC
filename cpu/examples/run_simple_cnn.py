"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""
import time
import os
from stensorflow.engine.start_server import start_local_server
import tensorflow as tf
start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))
from stensorflow.ml.nn.networks.SIMPLE_CNN import SIMPLE_CNN
from stensorflow.global_var import StfConfig
from cnn_utils import convert_datasets, load_data, calculate_score


def average_cnn_baseline(train_x, train_y, test_x, test_y):
    model = tf.keras.models.Sequential([
        # First Layer
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1), use_bias=False,
                               name="Conv"),
        tf.keras.layers.AvgPool2D(2, 2),
        tf.keras.layers.Flatten(),
        # Final Layer
        tf.keras.layers.Dense(10, name="Dense"),
        tf.keras.layers.Activation('softmax')
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    start_time = time.time()
    model.fit(train_x, train_y, epochs=10, batch_size=128)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # test result
    print("test result")
    pred_y = model.predict(test_x)
    # evaluate
    test_loss = model.evaluate(test_x, test_y)
    print("test result: " + str(test_loss))
    # accuracy = 0.8677
    # model.save("../output/simple_normal_mnist_model.h5")


def stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None):
    sess = tf.compat.v1.Session(StfConfig.target)
    Epochs = 10
    batch_size = 128
    learning_rate = 0.01
    record_num = train_x.shape[0]  # the num of train samples
    batch_num_per_epoch = record_num // batch_size
    train_batch_num = Epochs * batch_num_per_epoch
    # train_batch_num = 21
    print("train_batch_num: " + str(train_batch_num))
    pred_batch_num = test_x.shape[0] // batch_size
    predict_file_name = "../output/simple_mnist_predict.txt"

    x_train, y_train, x_test, y_test = convert_datasets(train_x=train_x, train_y=train_y,
                                                        test_x=test_x, test_y=test_y,
                                                        epoch=Epochs, batch_size=batch_size)
    model = SIMPLE_CNN(feature=x_train, label=y_train, loss="CrossEntropyLossWithSoftmax")
    model.compile()
    print("success compile")
    print("start train model")
    start_time = time.time()
    model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=0.0, sess=sess)
    end_time = time.time()
    print("train time=", end_time - start_time)
    # model.compare(keras_weight, sess=sess, model_file_machine='R')
    if keras_weight:
        # load weights
        print("start replace")
        model.replace_weight(keras_weight)
    print("start predict")
    model.predict_to_file(sess, x_test, predict_file_name, pred_batch_num=pred_batch_num,
                          model_file_machine='R', record_num_ceil_mod_batch_size=batch_size, with_sigmoid=False)
    # model.save_model(save_file_path="../output/STF_CNN.npz", sess=sess, model_file_machine='R')


if __name__ == "__main__":

    # compare_weight("../output/simple_normal_mnist_model.h5",
    #                "../output/STF_CNN.npz")
    # exit()

    train_x, train_y, test_x, test_y = load_data(normal=True, small=True)
    # compare_forward("../output/simple_normal_mnist_model.h5",
    #                 stf_predict_path="../output/simple_mnist_predict.txt",
    #                 test_x=test_x)
    # exit()

    # average_cnn_baseline(train_x, train_y, test_x, test_y)
    # exit()
    # res = np.load("../output/STF_CNN.npz", allow_pickle=True)
    # print(res)
    # res = res["weight"]
    # for i in range(len(res)):
    #     print(res[i].shape)
    #     print(res[i])
    # exit()
    # keras_model = tf.keras.models.load_model("../output/simple_normal_mnist_model.h5")
    # keras_weight = keras_model.get_weights()
    # test_loss = keras_model.evaluate(test_x, test_y)
    # print("keras test result: " + str(test_loss))
    # compare_dense_out(keras_model)
    # print(dense_out)

    # exit()
    stf_cnn_test(train_x, train_y, test_x, test_y, keras_weight=None)
    calculate_score("../output/simple_mnist_predict.txt")