import unittest
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedTensorBase, SharedPairBase
from stensorflow.basic.basic_class.share import SharedTensor
import numpy as np
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_to_compress_tensor(self):
        x = tf.constant(np.random.uniform(size=[4, 5, 6], low=0, high=7), dtype='int64')
        
        sx = SharedTensor(inner_value=x, module=7)
        cx=sx.to_compress_tensor()
        
        dcx= SharedTensor(module=7, shape=[4,5,6])
        dcx.decompress_from(cx, shape=[4,5,6])

        result = self.sess.run((dcx-sx).inner_value)
        self.assertLess(np.linalg.norm(result), 1E-3)


        
        
        


        






if __name__ == '__main__':
    unittest.main()
