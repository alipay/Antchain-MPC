import unittest
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedTensorBase, SharedPairBase
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_PrivateTensor, BM_PrivateTensor_SharedPair, \
    BM_SharedPair_PrivateTensor, BM_SharedPair_SharedPair
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_bm_PrivateTensor_PrivateTensor(self):
        x = PrivateTensorBase(owner="L")
        x.load_from_tf_tensor(tf.constant([3.45]*100))

        y = PrivateTensorBase(owner="R")
        y.load_from_tf_tensor(tf.constant([2.02]*100))

        z = BM_PrivateTensor_PrivateTensor(x, y, lambda x, y: x * y)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor("R"))[0], 3.45*2.02, delta=1E-3)

    def test_bm_PrivateTensor_PrivateTensor_PRF(self):
        x = PrivateTensorBase(owner="L")
        x.load_from_tf_tensor(tf.constant([3.45]*100))

        y = PrivateTensorBase(owner="R")
        y.load_from_tf_tensor(tf.constant([2.02]*100))

        z = BM_PrivateTensor_PrivateTensor(x, y, lambda a, b: a * b)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor("R"))[0], 3.45*2.02, delta=1E-3)

    def test_bm_PrivateTensor_SharedPair(self):
        x = PrivateTensorBase(owner="R")
        a = tf.constant([[-1.3, 2.0], [3.0, -1.0]])
        x.load_from_tf_tensor(a)

        u = tf.constant([[2, -1], [-2, 1]], 'int64')
        v = tf.constant([[2, -1], [-2, 1]], 'int64')
        c = u+v
        u = SharedTensorBase(u)
        v = SharedTensorBase(v)
        y = SharedPairBase(ownerL="L", ownerR="R", xL=u, xR=v, fixedpoint=0)

        z = BM_PrivateTensor_SharedPair(x, y, lambda x, y: x @ y)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertLess(self.sess.run(tf.norm(z.to_tf_tensor("R") - tf.cast(a@tf.cast(c,'float32'), 'float64'))) , 0.01)

    def test_bm_PrivateTensor_SharedPair_PRF(self):
        x = PrivateTensorBase(owner="R")
        a = tf.constant([[-1.3, 2.0], [3.0, -1.0]])
        x.load_from_tf_tensor(a)

        u = tf.constant([[2, -1], [-2, 1]], 'int64')
        v = tf.constant([[2, -1], [-2, 1]], 'int64')
        c = u + v
        u = SharedTensorBase(u)
        v = SharedTensorBase(v)
        y = SharedPairBase(ownerL="L", ownerR="R", xL=u, xR=v, fixedpoint=0)

        z = BM_PrivateTensor_SharedPair(x, y, lambda x, y: x @ y)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertLess(self.sess.run(tf.norm(z.to_tf_tensor("R") - tf.cast(a @ tf.cast(c, 'float32'), 'float64'))),
                        0.01)

    def test_bm_SharedPair_SharedPair(self):
        x = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor(tf.constant([3.45]*100))

        y = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        y.load_from_tf_tensor(tf.constant([-2.02]*100))

        z = BM_SharedPair_SharedPair(x, y, lambda a, b: a * b)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor("R"))[0], 3.45*(-2.02), delta=0.001)


    def test_bm_SharedPair_SharedPair_PRF(self):
        x = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        x.load_from_tf_tensor(tf.constant([3.45]*100))

        y = SharedPairBase(ownerL="L", ownerR="R", shape=[1])
        y.load_from_tf_tensor(tf.constant([-2.02]*100))

        z = BM_SharedPair_SharedPair(x, y, lambda a, b: a * b)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor("R"))[0], 3.45*(-2.02), delta=0.001)



if __name__ == '__main__':
    unittest.main()
