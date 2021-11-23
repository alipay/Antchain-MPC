import unittest
from stensorflow.basic.protocol.pm1_act import pm1_pair_act_pair, SharedTensorInt65, SharedPairInt65
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from stensorflow.global_var import StfConfig
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
import numpy as np
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_shared_pm1_act(self):
        aL = tf.constant([[1,0], [0,1]] , dtype='int64')
        aR = tf.constant([[0,0], [0,0]], dtype='int64')
        t = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, shape=[2,2])
        t.xL = SharedTensorBase(inner_value=aL, module=2)
        t.xR = SharedTensorBase(inner_value=aR, module=2)


        bL = tf.constant([[2,3],[6,1]], dtype='int64')
        bR = tf.constant([[1,1],[1,1]], dtype='int64')
        y = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, shape=[2,2])
        y.xL = SharedTensorBase(inner_value=bL)
        y.xR = SharedTensorBase(inner_value=bR)

        x_act_y = pm1_pair_act_pair(t, y, StfConfig.RS[0], prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.initialize_all_variables())
        z=self.sess.run(x_act_y.to_tf_tensor("R"))
        self.assertLess(np.linalg.norm(z-[[-3, 4], [7, -2]]), 1E-3)

    def test_shared_pm1_act_int65(self):
        aL = tf.constant([[1,0], [0,1]] , dtype='int64')
        aR = tf.constant([[1,1], [1,0]], dtype='int64')
        t = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, shape=[2,2])
        t.xL = SharedTensorBase(inner_value=aL, module=2)
        t.xR = SharedTensorBase(inner_value=aR, module=2)

        bL = tf.constant([[1<<62,1<<61],[6,1]], dtype='int64')
        bR = tf.constant([[1,1],[1,1]], dtype='int64')
        y = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, shape=[2,2])
        y.xL = SharedTensorBase(inner_value=bL)
        y.xR = SharedTensorBase(inner_value=bR)
        y = SharedPairInt65.from_SharedPair(y)

        x_act_y = pm1_pair_act_pair(t, y, StfConfig.RS[0], prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.initialize_all_variables())
        z=self.sess.run(x_act_y.to_SharedPair().to_tf_tensor("R"))
        w=np.power(-1, np.array([[0,1],[1,1]]))*np.array([[(1<<62)+1, (1<<61)+1],[7, 2]])
        self.assertLess(np.linalg.norm(z-w), 1E-3)

if __name__ == '__main__':
    unittest.main()
