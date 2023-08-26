import unittest
from stensorflow.basic.basic_class.base import SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.protocol.fnz import fnz_v1, fnz_v2, fnz_v3

import tensorflow as tf
import numpy as np
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_fnz(self):
        x = SharedTensorBase(inner_value=tf.constant([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]], dtype='int64'),
                             module=7)
        xL = x.random_uniform_adjoint()
        xR = x - xL
        x = SharedPair(ownerL="L", ownerR="R", xL=xL, xR=xR, fixedpoint=0)
        i = fnz_v1(x, prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        result = self.sess.run(i.to_tf_tensor("R"))
        #print(result)
        self.assertLess(np.sum(np.power(result - [1, 0, 3], 2)), 1E-3)

    def test_fnz2(self):
        x = SharedTensorBase(inner_value=tf.constant([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]], dtype='int64'),
                             module=7)
        xL = x.random_uniform_adjoint()
        xR = x - xL
        x = SharedPair(ownerL="L", ownerR="R", xL=xL, xR=xR, fixedpoint=0)
        i = fnz_v2(x, prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        result = self.sess.run(i.to_tf_tensor("R"))
        #print(result)
        self.assertLess(np.sum(np.power(result - [1, 0, 3], 2)), 1E-3)

    def test_fnz3(self):
        x = SharedTensorBase(inner_value=tf.constant([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]], dtype='int64'),
                             module=7)
        xL = x.random_uniform_adjoint()
        xR = x - xL
        x = SharedPair(ownerL="L", ownerR="R", xL=xL, xR=xR, fixedpoint=0)
        i = fnz_v3(x, prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        result = self.sess.run(i.to_tf_tensor("R"))
        #print(result)
        self.assertLess(np.sum(np.power(result - [1, 0, 3], 2)), 1E-3)


if __name__ == '__main__':
    unittest.main()
