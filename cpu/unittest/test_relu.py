import unittest
from stensorflow.basic.basic_class.pair import SharedPair, SharedPairBase, SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor
import tensorflow as tf
import numpy as np
import datetime
from stensorflow.global_var import StfConfig
from stensorflow.basic.operator.order import is_not_negative, is_positive
from stensorflow.basic.operator.relu import relu, drelu_binary_const, drelu_binary_linear, drelu_binary_log
from stensorflow.random.random import random_init
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()





    def test_drelu_binary_high_delay(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
        y = drelu_binary_const(x)
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result = self.sess.run(y)
        b = (a>=0)
        self.assertLess(np.linalg.norm(result-b), 1E-3)

    def test_drelu_binary_middle_delay(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
        y = drelu_binary_log(x)
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result = self.sess.run(y)
        b = (a>=0)
        self.assertLess(np.linalg.norm(result-b), 1E-3)

    def test_drelu_binary_low_delay(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
        y = drelu_binary_linear(x)
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result = self.sess.run(y)
        b = (a>=0)
        self.assertLess(np.linalg.norm(result-b), 1E-3)

    def test_relu(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
        y = relu(x)
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result = self.sess.run(y)
        b = a*(a>0)
        self.assertLess(np.linalg.norm(result-b), 1E-3)







if __name__ == '__main__':
    unittest.main()
