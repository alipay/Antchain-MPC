import unittest
from stensorflow.basic.protocol.module_transform import module_transform,\
    module_transform_withPRF
import numpy as np
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase
from stensorflow.global_var import StfConfig
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_module_transform(self):
        with tf.device(StfConfig.workerL[0]):
            a = np.random.random_integers(low=0, high=1, size=[8])

            x = SharedTensorBase(module=2, inner_value=tf.constant(a, dtype='int64'))

        with tf.device(StfConfig.workerR[0]):
            b = np.random.random_integers(low=0, high=1, size=[8])

            y = SharedTensorBase(module=2, inner_value=tf.constant(b, dtype='int64'))

        z = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, xL=x, xR=y)

        w = module_transform(z, new_module=7, compress_flag=False)
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.assertLess(np.sum(np.power(self.sess.run(w.to_tf_tensor("R"))-(a+b)%2, 2)), 1E-3)


    def test_module_transform_compress(self):
        with tf.device(StfConfig.workerL[0]):
            a = np.random.random_integers(low=0, high=1, size=[8])

            x = SharedTensorBase(module=2, inner_value=tf.constant(a, dtype='int64'))

        with tf.device(StfConfig.workerR[0]):
            b = np.random.random_integers(low=0, high=1, size=[8])

            y = SharedTensorBase(module=2, inner_value=tf.constant(b, dtype='int64'))

        z = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, xL=x, xR=y)


        w = module_transform(z, new_module=7, compress_flag=True)
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.assertLess(np.sum(np.power(self.sess.run(w.to_tf_tensor("R"))-(a+b)%2, 2)), 1E-3)




    def test_module_transform_withPRF(self):
        with tf.device(StfConfig.workerL[0]):
            a = np.random.random_integers(low=0, high=1, size=[8])

            x = SharedTensorBase(module=2, inner_value=tf.constant(a, dtype='int64'))

        with tf.device(StfConfig.workerR[0]):
            b = np.random.random_integers(low=0, high=1, size=[8])

            y = SharedTensorBase(module=2, inner_value=tf.constant(b, dtype='int64'))

        z = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, xL=x, xR=y)

        w = module_transform_withPRF(z, new_module=7, compress_flag=False)
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.assertLess(np.sum(np.power(self.sess.run(w.to_tf_tensor("R"))-(a+b)%2, 2)), 1E-3)



    def test_module_transform_compress_withPRF(self):
        with tf.device(StfConfig.workerL[0]):
            a = np.random.random_integers(low=0, high=1, size=[8])

            x = SharedTensorBase(module=2, inner_value=tf.constant(a, dtype='int64'))

        with tf.device(StfConfig.workerR[0]):
            b = np.random.random_integers(low=0, high=1, size=[8])

            y = SharedTensorBase(module=2, inner_value=tf.constant(b, dtype='int64'))

        z = SharedPairBase(ownerL="L", ownerR="R", fixedpoint=0, xL=x, xR=y)


        w = module_transform_withPRF(z, new_module=7, compress_flag=True)
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.assertLess(np.sum(np.power(self.sess.run(w.to_tf_tensor("R"))-(a+b)%2, 2)), 1E-3)









if __name__ == '__main__':
    unittest.main()
