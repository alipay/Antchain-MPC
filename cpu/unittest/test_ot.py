import unittest
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedTensorBase, SharedPairBase
from stensorflow.basic.protocol.ot import assistant_ot

import tensorflow as tf
import numpy as np
import time
from stensorflow.engine.start_server import start_local_server
tf.compat.v1.disable_eager_execution()

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()


    def test_assistant_ot(self):
        x = tf.random.uniform(shape=[100, 8, 4])
        i = tf.random.uniform(shape=[100, 8], minval=0, maxval=4, dtype='int64')

        xx = PrivateTensorBase(owner="L")
        xx.load_from_tf_tensor(x)
        ii = PrivateTensorBase(owner="R", fixedpoint=0, module=4,
                              inner_value=i)

        z = assistant_ot(xx, ii, prf_flag=True, compress_flag=True)
        y = tf.gather(params=x, indices=i, axis=len(i.shape),
                  batch_dims=len(i.shape))
        self.sess.run(tf.compat.v1.initialize_all_variables())
        result = self.sess.run(z.to_tf_tensor("R")-tf.cast(y, 'float64'))
        self.assertLess(np.sum(np.abs(result)), 1E-1)


if __name__ == '__main__':
    unittest.main()
