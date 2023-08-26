import unittest

import numpy as np
from stensorflow.basic.basic_class.pair import SharedTensorBase,SharedPairBase
from stensorflow.basic.basic_class.private import PrivateTensor
import tensorflow as tf
from stensorflow.basic.operator.selectshare import native_select, select_share
from stensorflow.global_var import StfConfig
from stensorflow.random.random import random_init
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_native_select(self):
        with tf.device(StfConfig.workerL[0]):
            tL = np.random.randint(low=0, high=2, size=[32,8])
            tL = tf.constant(tL)
            tL = SharedTensorBase(inner_value=tL, module=2)
            xL = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xL = tf.constant(xL, dtype='int64')
            xL = SharedTensorBase(inner_value=xL)

        with tf.device(StfConfig.workerR[0]):
            tR = np.random.randint(low=0, high=2, size=[32,8])
            tR = tf.constant(tR)
            tR = SharedTensorBase(inner_value=tR, module=2)
            xR = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xR = tf.constant(xR, dtype='int64')
            xR = SharedTensorBase(inner_value=xR)

        t = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=tL, xR=tR, fixedpoint=0)

        x = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=xL, xR=xR, fixedpoint=0)

        tx = native_select(t, x, prf_flag=True, compress_flag=True)

        z = tx.to_tf_tensor("R")-t.to_tf_tensor("R")*x.to_tf_tensor("R")
        self.sess.run(random_init())
        self.assertEqual(np.count_nonzero(self.sess.run(z)), 0)
        

    def test_select_using_pm1_act(self):
        with tf.device(StfConfig.workerL[0]):
            tL = np.random.randint(low=0, high=2, size=[32,8])
            tL = tf.constant(tL)
            tL = SharedTensorBase(inner_value=tL, module=2)
            xL = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xL = tf.constant(xL, dtype='int64')
            xL = SharedTensorBase(inner_value=xL)

        with tf.device(StfConfig.workerR[0]):
            tR = np.random.randint(low=0, high=2, size=[32,8])
            tR = tf.constant(tR)
            tR = SharedTensorBase(inner_value=tR, module=2)
            xR = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xR = tf.constant(xR, dtype='int64')
            xR = SharedTensorBase(inner_value=xR)

        t = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=tL, xR=tR, fixedpoint=0)

        x = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=xL, xR=xR, fixedpoint=0)

        tx = select_share(t, x, prf_flag=True, compress_flag=True)

        z = tx.to_tf_tensor("R")-t.to_tf_tensor("R")*x.to_tf_tensor("R")
        self.sess.run(random_init())
        self.assertEqual(np.count_nonzero(self.sess.run(z)), 0)



    def test_select_Private_Private_share(self):
        with tf.device(StfConfig.workerL[0]):
            tL = np.random.randint(low=0, high=2, size=[32,8])
            tL = tf.constant(tL)
            tL = PrivateTensor(inner_value=tL, module=2, fixedpoint=0, owner=StfConfig.workerL[0])


        with tf.device(StfConfig.workerR[0]):
            xR = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xR = tf.constant(xR, dtype='int64')
            xR = PrivateTensor(inner_value=xR, owner=StfConfig.workerR[0])

        tx = select_share(tL, xR, prf_flag=True, compress_flag=True, )

        z = tx.to_tf_tensor("R")-tL.to_tf_tensor("R")*xR.to_tf_tensor("R")
        self.sess.run(random_init())
        self.assertEqual(np.count_nonzero(self.sess.run(z)), 0)


    def test_select_Private_SharedPair_share(self):
        with tf.device(StfConfig.workerL[0]):
            tL = np.random.randint(low=0, high=2, size=[32,8])
            tL = tf.constant(tL)
            tL = PrivateTensor(inner_value=tL, module=2, fixedpoint=0, owner=StfConfig.workerL[0])
            xL = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xL = tf.constant(xL, dtype='int64')
            xL = SharedTensorBase(inner_value=xL)

        with tf.device(StfConfig.workerR[0]):
            # tR = np.random.randint(low=0, high=2, size=[32,8])
            # tR = tf.constant(tR)
            # tR = SharedTensorBase(inner_value=tR, module=2)
            xR = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xR = tf.constant(xR, dtype='int64')
            xR = SharedTensorBase(inner_value=xR)

        # t = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=tL, xR=tR, fixedpoint=0)

        x = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=xL, xR=xR, fixedpoint=0)

        tx = select_share(tL, x, prf_flag=True, compress_flag=True, )

        z = tx.to_tf_tensor("R")-tL.to_tf_tensor("R")*x.to_tf_tensor("R")
        self.sess.run(random_init())
        self.assertEqual(np.count_nonzero(self.sess.run(z)), 0)


    def test_select_SharedPair_Private(self):
        with tf.device(StfConfig.workerL[0]):
            tL = np.random.randint(low=0, high=2, size=[32,8])
            tL = tf.constant(tL)
            tL = SharedTensorBase(inner_value=tL, module=2)
            # xL = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            # xL = tf.constant(xL, dtype='int64')
            # xL = SharedTensorBase(inner_value=xL)

        with tf.device(StfConfig.workerR[0]):
            tR = np.random.randint(low=0, high=2, size=[32,8])
            tR = tf.constant(tR)
            tR = SharedTensorBase(inner_value=tR, module=2)
            xR = np.random.randint(low=-(1<<62), high=(1<<62), size=[32,8])
            xR = tf.constant(xR, dtype='int64')
            xR = PrivateTensor(inner_value=xR, fixedpoint=0, owner=StfConfig.workerR[0])

        t = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=tL, xR=tR, fixedpoint=0)

        #x = SharedPairBase(ownerL=StfConfig.workerL[0], ownerR=StfConfig.workerR[0], xL=xL, xR=xR, fixedpoint=0)

        tx = select_share(t, xR, prf_flag=False, compress_flag=False)

        z = tx.to_tf_tensor("R")-t.to_tf_tensor("R")*xR.to_tf_tensor("R")
        self.sess.run(random_init())
        self.assertEqual(np.count_nonzero(self.sess.run(z)), 0)

if __name__ == '__main__':
    unittest.main()
