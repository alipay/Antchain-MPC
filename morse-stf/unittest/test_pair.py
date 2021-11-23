import unittest
from stensorflow.basic.basic_class.pair import SharedPair, SharedVariablePair
import tensorflow as tf
from stensorflow.basic.basic_class.base import PrivateTensorBase
from stensorflow.basic.operator.algebra import concat
from stensorflow.random.random import random_init
import numpy as np
from stensorflow.engine.start_server import start_local_server
tf.compat.v1.disable_eager_execution()

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_SharedPair(self):
        x = SharedPair(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor([12.34])
        y = 2 + x
        print(y)
        z = y.to_tf_tensor(owner="L")
        random_init(self.sess)
        self.assertLess(np.abs(self.sess.run(z)-14.34), 1E-3)



    def test_concat(self):

        a = 12.34
        b = a ** 3
        x = SharedPair(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor([a])
        y = x ** 3
        z = concat([x, y], axis=0)
        random_init(self.sess)
        z = self.sess.run(z.to_tf_tensor("R"))
        self.assertLess(np.sum(np.abs(z-[a,b])), 1E-1)


    def test_concat2(self):
        x = SharedPair(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor([12.34])
        y = PrivateTensorBase("L")
        y.load_from_tf_tensor([12.34])
        z = concat([x,y], axis=0)
        random_init(self.sess)
        z = self.sess.run(z.to_tf_tensor("R"))
        self.assertLess(np.sum(np.abs(z-[12.34, 12.34])), 1E-1)


if __name__ == '__main__':
    unittest.main()
