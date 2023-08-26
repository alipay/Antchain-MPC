import unittest
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedTensorBase, SharedPairBase
from stensorflow.basic.operator.algebra import concat
import numpy as np
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
tf.compat.v1.disable_eager_execution()
import os
start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))



class MyTestCase(unittest.TestCase):


    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()


    def test_PrivateTensorBase0(self):
        px = PrivateTensorBase(owner="R", fixedpoint=-3, inner_value=5)
        print(self.sess.run(px.to_tf_tensor("R")))

    def test_PrivateTensorBase(self):
        x = tf.constant([12.34])
        px = PrivateTensorBase("L:0")
        px.load_from_tf_tensor(x)

        y = tf.constant([-12.34])
        py = PrivateTensorBase("L")
        py.load_from_tf_tensor(y)

        py.dup_with_precision(3)

        
        # self.assertEqual()
        self.assertAlmostEqual(self.sess.run(py.to_tf_tensor()), self.sess.run(y), delta=0.1)
        z = px.zeros_like()
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor()), 0.0, delta=0.001)
        z = px.reshape([1, 1, -1])
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor()), self.sess.run(px.to_tf_tensor()), delta=0.001)
        z = px.ones_like()
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor()), 1.0, delta=0.001)
        z = px.concat(other=py, axis=0)
        self.assertAlmostEqual(self.sess.run(tf.norm(z.to_tf_tensor() - [12.34, -12.34])), 0, delta=0.001)
        z = px.identity()
        self.assertEqual(self.sess.run(z.to_tf_tensor()), self.sess.run(px.to_tf_tensor()))
        z = px.slice(begin=[0], size=[1])
        self.assertEqual(self.sess.run(z.to_tf_tensor()), self.sess.run(px.to_tf_tensor()))
        z = px.split(size_splits=[0, 1], axis=0)
        self.assertEqual(self.sess.run(z[1].to_tf_tensor()), self.sess.run(px.to_tf_tensor()))
        z = px.to_SharedTensor()
        self.assertEqual(self.sess.run(z.inner_value), self.sess.run(px.inner_value))
        z = px.stack(other=px, axis=0)
        self.assertAlmostEqual(
            self.sess.run(tf.norm(z.to_tf_tensor() - tf.stack([px.to_tf_tensor(), px.to_tf_tensor()], axis=0))),
            0.0, delta=0.001)
        z = px.dup_with_precision(new_fixedpoint=5)
        self.assertAlmostEqual(self.sess.run(z.to_tf_tensor()), self.sess.run(px.to_tf_tensor()), delta=0.1)


    def test_SharedPairBase(self):
        x = tf.constant([12.34])
        px = PrivateTensorBase("L:0")
        px.load_from_tf_tensor(x)
        sx = px.to_SharedTensor()

        

        y = tf.constant([-7.56])

        py = PrivateTensorBase("R")
        

        py.load_from_tf_tensor(y)
        

        sy = py.to_SharedTensor()
        

        z = SharedPairBase(xL=sx, xR=sy, ownerL="L", ownerR="R")
        

        z1 = z.dup_with_precision(7)
        # self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
        self.assertLess(self.sess.run(tf.norm(z.to_tf_tensor("R") - z1.to_tf_tensor("R"))), 0.01)

    def test_cumulative_sum(self):
        x = SharedTensorBase(inner_value=tf.constant([[1, 2, 3], [4, 5, 6]]), module=7)
        y = x.cumulative_sum(axis=-1)
        z = x.cumulative_sum(axis=0)
        # with tf.compat.v1.Session("grpc://0.0.0.0:8887") as self.sess:
        y = self.sess.run(y.inner_value)
        z = self.sess.run(z.inner_value)
        
        
        self.assertTrue((y == [[1, 3, 6], [4, 2, 1]]).all())
        self.assertTrue((z == [[1, 2, 3], [5, 0, 2]]).all())

    def test_concat(self):
        x = tf.constant([12.34])
        px = PrivateTensorBase("L:0")
        px.load_from_tf_tensor(x)
        
        

        y = tf.constant([-12.34])
        py = PrivateTensorBase("L")
        py.load_from_tf_tensor(y)

        pz = concat([px, py], axis=0)
        # self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
        self.assertLess(self.sess.run(tf.norm(pz.to_tf_tensor() - [12.34, -12.34])), 0.001)

    def test_to_compress_tensor(self):
        x = tf.constant(np.random.uniform(size=[4, 5, 6], low=0, high=16), dtype='int64')
        
        sx = SharedTensorBase(inner_value=x, module=16)
        cx = sx.to_compress_tensor()
        
        dcx = SharedTensorBase(module=16, shape=[4, 5, 6])
        dcx.decompress_from(cx, shape=[4, 5, 6])
        # self.sess = tf.compat.v1.Session('grpc://0.0.0.0:8887')
        self.assertLess(self.sess.run(tf.reduce_sum(tf.pow(dcx.inner_value - x, 2))), 1E-8)



if __name__ == '__main__':
    unittest.main()
