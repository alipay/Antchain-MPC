import unittest

from stensorflow.basic.protocol.compare import less_SharedTensorBase_SharedTensorBase, greater_SharedTensorBase_SharedTensorBase
from stensorflow.basic.basic_class.base import PrivateTensorBase
import tensorflow as tf
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_less(self):
        a = np.random.random_integers(low=0, high=1, size=[3, 64])
        
        x = PrivateTensorBase(owner="L", module=2, inner_value=tf.constant(a, dtype='int64'))

        b = np.random.random_integers(low=0, high=1, size=[3, 64])
        
        y = PrivateTensorBase(owner="R", module=2, inner_value=tf.constant(b, dtype='int64'))

        xly = less_SharedTensorBase_SharedTensorBase(x.to_SharedTensor(), y.to_SharedTensor(), x.owner, y.owner)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        xly_real = list(int(pair[0]<pair[1]) for pair in zip(a.tolist(), b.tolist()))
        self.assertListEqual(self.sess.run(xly.to_tf_tensor("L")).tolist(), xly_real)


    def test_grether(self):

        a = np.random.random_integers(low=0, high=1, size=[3, 64])
        
        x = PrivateTensorBase(owner="L", module=2, inner_value=tf.constant(a, dtype='int64'))

        b = np.random.random_integers(low=0, high=1, size=[3,64])
        
        y = PrivateTensorBase(owner="R", module=2, inner_value=tf.constant(b, dtype='int64'))

        xgy=greater_SharedTensorBase_SharedTensorBase(x.to_SharedTensor(), y.to_SharedTensor(),x.owner, y.owner)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        xgy_real = list(int(pair[0]>pair[1]) for pair in zip(a.tolist(), b.tolist()))
        self.assertListEqual(self.sess.run(xgy.to_tf_tensor("L")).tolist(), xgy_real)


if __name__ == '__main__':
    unittest.main()
