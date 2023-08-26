import unittest
from stensorflow.basic.basic_class.bitwise import SharedPairBitwise, SharedTensorBitwise, PrivateTensorBitwise
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
    def test_SharedPairBitwise(self):
        x = SharedPairBitwise(ownerL="L", ownerR="R", xL=SharedTensorBitwise(inner_value=tf.constant([0,1], dtype='int64')),
                              xR=SharedTensorBitwise(inner_value=tf.constant([1, 0], dtype='int64')))

        
        y = SharedPairBitwise(ownerL="L", ownerR="R", xL=SharedTensorBitwise(inner_value=tf.constant([0,1], dtype='int64')),
                              xR=SharedTensorBitwise(inner_value=tf.constant([1, 0], dtype='int64')))
        
        z = x*y
        
        w = x + y
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertListEqual(list(self.sess.run(z.to_tf_tensor())), [1, 1])
        self.assertListEqual(list(self.sess.run(w.to_tf_tensor())), [0, 0])


if __name__ == '__main__':
    unittest.main()
