import unittest
from stensorflow.ml.nn.networks.DNN_with_SL import DNN_with_SL2, DNN_with_SL
from stensorflow.ml.nn.networks.DNN import DNN
import tensorflow as tf
from stensorflow.basic.basic_class.private import PrivateTensor
import time
from stensorflow.engine.start_server import start_local_server

import os

stf_home = os.environ.get("stf_home", "..")
start_local_server(os.path.join(stf_home, "conf", "config.json"))
predict_file_name = "/dev/null"

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_DNN_train_and_predict_secure_level2(self):
        featureNumL = 5
        featureNumR = 5
        record_num = 6
        epoch = 1
        batch_size = 3
        l2_regularization = 0.0
        batch_num_per_epoch = record_num // batch_size
        train_batch_num = epoch * batch_num_per_epoch + 1
        learning_rate = 0.01

        xL_train = PrivateTensor(owner='L')
        xyR_train = PrivateTensor(owner='R')
        xL_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumL]))
        xyR_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumR + 1]))
        xR_train, y_train = xyR_train.split(size_splits=[featureNumR, 1], axis=1)
        num_features = featureNumL + featureNumR
        model = DNN_with_SL2(feature=xL_train, label=y_train, dense_dims=[num_features, 3, 3, 1],
                             feature_another=xR_train, secure_config=[2, 1, 1])
        model.compile()
        start_time = time.time()
        model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization,
                        sess=self.sess)

        end_time = time.time()
        print("train time=", end_time - start_time)
        xL_test = PrivateTensor(owner='L')
        xRy_test = PrivateTensor(owner='R')
        xL_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumL]))
        id = xRy_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumR + 1]))
        xR_test, y_test = xRy_test.split(size_splits=[-1, 1], axis=1)

        model.predict_to_file(self.sess, x=xL_test, x_another=xR_test, predict_file_name=predict_file_name,
                              batch_num=1, idx=id,
                              model_file_machine='R',
                              record_num_ceil_mod_batch_size=3)

    def test_DNN_train_and_predict_secure_level(self):
        featureNumX = 5
        featureNumY = 0
        record_num = 6
        epoch = 1
        batch_size = 3
        learning_rate = 0.01
        l2_regularization = 0.0
        train_batch_num = epoch * record_num // batch_size + 1

        x_train = PrivateTensor(owner='L')
        y_train = PrivateTensor(owner='R')

        x_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        y_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY + 1]))

        num_features = featureNumX + featureNumY
        model = DNN_with_SL(feature=x_train, label=y_train, dense_dims=[num_features, 3, 3, 1], secure_config=[1, 2, 1])
        model.compile()

        start_time = time.time()
        model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization,
                        sess=self.sess)

        end_time = time.time()
        print("train time=", end_time - start_time)

        x_test = PrivateTensor(owner='L')
        y_test = PrivateTensor(owner='R')

        x_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        id = y_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY + 1]))

        model.predict_to_file(self.sess, x_test, predict_file_name, batch_num=1, idx=id,
                              model_file_machine='R',
                              record_num_ceil_mod_batch_size=3)

    def test_DNN_train_and_predict(self):
        featureNumX = 5
        featureNumY = 0
        record_num = 6

        epoch = 1
        batch_size = 3
        learning_rate = 0.01
        l2_regularization = 0.0

        train_batch_num = epoch * record_num // batch_size + 1

        x_train = PrivateTensor(owner='L')
        y_train = PrivateTensor(owner='R')

        x_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        y_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY + 1]))

        num_features = featureNumX + featureNumY
        model = DNN(feature=x_train, label=y_train, dense_dims=[num_features, 3, 3, 1])
        model.compile()

        start_time = time.time()
        model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization,
                        sess=self.sess)
        end_time = time.time()

        print("train time=", end_time - start_time)
        x_test = PrivateTensor(owner='L')
        y_test = PrivateTensor(owner='R')

        x_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        id = y_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY + 1]))

        model.predict_to_file(self.sess, x_test, predict_file_name, batch_num=1, idx=id,
                              model_file_machine='R',
                              record_num_ceil_mod_batch_size=3)

    def test_DNN_train_and_predict2(self):
        featureNumL= 5
        featureNumR = 5
        record_num = 6
        epoch = 1
        batch_size = 3
        l2_regularization = 0.0

        batch_num_per_epoch = record_num // batch_size
        train_batch_num = epoch * batch_num_per_epoch + 1

        learning_rate = 0.01

        xL_train = PrivateTensor(owner='L')
        xyR_train = PrivateTensor(owner='R')

        xL_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumL]))
        xyR_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumR + 1]))

        xR_train, y_train = xyR_train.split(size_splits=[featureNumR, 1], axis=1)

        num_features = featureNumL + featureNumR

        model = DNN(feature=xL_train, label=y_train, dense_dims=[num_features, 3, 3, 1],
                    feature_another=xR_train)
        model.compile()

        start_time = time.time()
        model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization,
                        sess=self.sess)
        end_time = time.time()
        print("train time=", end_time - start_time)

        xL_test = PrivateTensor(owner='L')
        xRy_test = PrivateTensor(owner='R')

        xL_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumL]))
        id = xRy_test.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumR + 1]))

        xR_test, y_test = xRy_test.split(size_splits=[-1, 1], axis=1)
        model.predict_to_file(self.sess, xL_test, predict_file_name, x_another=xR_test, batch_num=1, idx=id,
                              model_file_machine='R',
                              record_num_ceil_mod_batch_size=3)


if __name__ == '__main__':
    unittest.main()
