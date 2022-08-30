import unittest
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, PrivateTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.basic.operator.arithmetic import add, sub, matmul, mul
import tensorflow as tf
from stensorflow.random.random import random_init
import stensorflow as stf
import numpy as np
import os
from stensorflow.engine.start_server import start_local_server

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_arithmetic(self):
        a = -12.34
        b = 7.65
        x = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor(tf.constant(a))

        y = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        y.load_from_tf_tensor(tf.constant(b))

        z = add(x, y).to_tf_tensor(owner="L")
        w = sub(x, y).to_tf_tensor(owner="L")
        u = mul(x, y).to_tf_tensor(owner="L")
        v = matmul(x.reshape([1, 1]), y.reshape([1, 1])).to_tf_tensor(owner="R")
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertAlmostEqual(self.sess.run(z), a + b, delta=0.001)
        self.assertAlmostEqual(self.sess.run(w), a - b, delta=0.001)
        self.assertAlmostEqual(self.sess.run(u), a * b, delta=0.001)
        self.assertAlmostEqual(self.sess.run(v), a * b, delta=0.001)


    def test_mul(self):
        tf.compat.v1.disable_eager_execution()

        x_test = PrivateTensor(owner='L')
        x_test.load_from_tf_tensor(tf.constant([[1.0]*1000000]))
        x_test.load_from_numpy()

        y_test = PrivateTensor(owner='R')
        y_test.load_from_tf_tensor(tf.constant([[1.0]*1000000]))
        z = x_test * y_test

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.run(tf.norm(z.to_tf_tensor("L")))


    def test_matmul(self):
        tf.compat.v1.disable_eager_execution()

        x_test = PrivateTensor(owner='L')
        x_test.load_from_tf_tensor(tf.constant([[0.1, 0.2], [2, 1]]))

        w_test = SharedVariablePair(ownerL="L", ownerR="R", shape=[2, 2])
        w_test.load_from_tf_tensor(tf.constant([[1, 2], [-1, 3]]))

        x_test.expend_dims(axis=2) - w_test.expend_dims(axis=0)

        b_test = SharedVariablePair(ownerL="L", ownerR="R", shape=[2, 1])
        b_test.load_from_tf_tensor(tf.constant([[0.2], [-0.2]]))

        y = x_test @ w_test
        z = y + b_test

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        self.assertLess(self.sess.run(tf.norm(z.to_tf_tensor("L") - [[0.1], [-0.3]])), 0.001)

    def test_inverse(self):
        a = np.random.uniform(low=0.0, high=100.0, size=[3, 3])
        print(a)
        b = 1/a
        a = tf.constant(a, dtype='float64')
        x = SharedPair(ownerL="L", ownerR="R", shape=[3, 3])
        x.load_from_tf_tensor(a)
        y = ~x
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result = self.sess.run([y])
        print(b)
        print(result)






if __name__ == '__main__':
    unittest.main()
