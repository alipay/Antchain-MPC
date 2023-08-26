import unittest

#from stensorflow.basic.protocol.compare import less_SharedTensorBase_SharedTensorBase, greater_SharedTensorBase_SharedTensorBase, greater_SharedTensorBase_SharedTensorBase2
from stensorflow.engine.float_compute_narrow import geq, less
from stensorflow.basic.basic_class.base import PrivateTensorBase
import tensorflow as tf
import numpy as np
from stensorflow.random.random import random_init
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_less(self):

        a = np.random.random_integers(low=0, high=1<<63-1, size=[3,3])
        
        x = PrivateTensorBase(owner="L", inner_value=tf.constant(a, dtype='int64'))

        b = np.random.random_integers(low=0, high=1<<63-1, size=[3,3])
        
        y = PrivateTensorBase(owner="R",inner_value=tf.constant(b,dtype='int64'))


        xly=less(x.inner_value, y.inner_value, x.owner, y.owner)

        self.sess.run(random_init())

        self.assertAlmostEqual(np.sum(np.abs(self.sess.run(xly.to_tf_tensor("R"))-(a<b))),0.0)  #4s408ms

            

if __name__ == '__main__':
    unittest.main()
