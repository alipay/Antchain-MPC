import unittest
from stensorflow.engine.start_server import start_local_server
import tensorflow as tf
import stensorflow as stf
from stensorflow.random.random import random_init
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.ml.logistic_regression import LogisticRegression
from stensorflow.ml.logistic_regression2 import LogisticRegression2

import os

stf_home = os.environ.get("stf_home", "..")
start_local_server(os.path.join(stf_home, "conf", "config.json"))
model_file_path = "/dev/null"

class MyTestCase(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_lr(self):

        featureNumX = 5
        featureNumY = 0
        record_num = 6
        epoch = 1
        batch_size = 3
        learning_rate = 0.01
        train_batch_num = epoch * record_num // batch_size + 1
        x_train = stf.PrivateTensor(owner='L')
        y_train = stf.PrivateTensor(owner='R')
        x_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        y_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY + 1]))
        model = LogisticRegression(num_features=featureNumX + featureNumY, learning_rate=learning_rate)
        model.fit(self.sess, x_train, y_train, num_batches=train_batch_num)
        model.save(model_file_path=model_file_path)

    def test_lr2(self):
        featureNumX = 5
        featureNumY = 5
        record_num = 6
        epoch = 1
        batch_size = 3
        learning_rate = 0.01

        train_batch_num = epoch * record_num // batch_size + 1

        xL_train = PrivateTensor(owner='L')
        xy_train = PrivateTensor(owner='R')

        xL_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumX]))
        xy_train.load_from_tf_tensor(tf.random.normal(shape=[batch_size, featureNumY+1]))
        xR_train, y_train = xy_train.split(size_splits=[featureNumY, 1], axis=1)
        model = LogisticRegression2(num_features_L=featureNumX, num_features_R=featureNumY, learning_rate=learning_rate,
                                    l2_regularzation=0.01)
        model.fit(sess=self.sess, x_L=xL_train, x_R=xR_train, y=y_train, num_batches=train_batch_num)
        model.save(model_file_path=os.path.join(stf_home, "output", "model"))

if __name__ == '__main__':
    unittest.main()
