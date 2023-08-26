import unittest
from stensorflow.basic.basic_class.pair import SharedPair, SharedPairBase, SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor
import tensorflow as tf
import numpy as np
import datetime
from stensorflow.global_var import StfConfig
from stensorflow.basic.operator.order import is_not_negative, is_positive
from stensorflow.basic.protocol.msb import msb, msb_log_round, special_mul
from stensorflow.random.random import random_init
from functools import reduce
from stensorflow.engine.start_server import start_local_server

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_msb(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(a)
        y = msb(x)
        y = y.to_tf_tensor("R")
        self.sess.run(random_init())
        result = self.sess.run(y)
        self.assertEqual(np.count_nonzero(result-(a<0)), 0)

    def test_msb2(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[10, 5])
        x.load_from_tf_tensor(a)
        y = msb_log_round(x).to_tf_tensor("R")
        self.sess.run(random_init())
        result = self.sess.run(y)
        self.assertEqual(np.count_nonzero(result-(a<0)), 0)


if __name__ == '__main__':
    unittest.main()
