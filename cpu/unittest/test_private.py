import unittest
import tensorflow as tf
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateVariable
from stensorflow.basic.basic_class.base import PrivateTensorBase
import numpy as np
from stensorflow.random.random import random_init
from stensorflow.engine.start_server import start_local_server

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_privateTensor(self):
        x = tf.constant([10.00])
        y = tf.constant([-5.67])

        px = PrivateTensor("L")
        py = PrivateTensor("R")
        px.load_from_tf_tensor(x)
        py.load_from_tf_tensor(y)


        z = x + y
        pz = px + py
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(z), delta=1E-4)

        z = x - y
        pz = px - py
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(z), delta=1E-4)

        z = x * y
        pz = px * py
        random_init(self.sess)
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(z), delta=1E-2)

        pw = pz.dup_with_precision(new_fixedpoint=7)
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(pw.to_tf_tensor("R")), delta=1E-2)

        pw = pz.stack(pz, axis=0)
        w = tf.stack([z, z], axis=0)
        self.assertLess(np.sum(np.abs((self.sess.run(pw.to_tf_tensor("R")) - self.sess.run(w)))), 1E-3)

        pw = pz.identity()
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(pw.to_tf_tensor("R")), delta=1E-4)

        pw = pz.ones_like()
        w = tf.ones_like(z)
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-4)

        pw = pz.to_private("R")
        self.assertAlmostEqual(self.sess.run(pz.to_tf_tensor("R")), self.sess.run(pw.to_tf_tensor("R")), delta=1E-4)

        pw = pz.zeros_like()
        w = tf.zeros_like(z)
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-4)

        pw = pz.reshape(shape=[-1])
        w = tf.reshape(z, shape=[-1])
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-3)

        pw = pz.squeeze(axis=0)
        w = tf.squeeze(z, axis=0)
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-3)

        pw = pz.transpose()
        w = tf.transpose(z)
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-3)

        pw = pz.lower_bound_like()
        self.assertLess(self.sess.run(pw.to_tf_tensor("R")), np.random.uniform())

        pw = pz.upper_bound_like()
        self.assertLess(np.random.uniform(), self.sess.run(pw.to_tf_tensor("R")))

        pw = pz.expend_dims(axis=1)
        w = tf.expand_dims(z, axis=1)
        self.assertAlmostEqual(self.sess.run(pw.to_tf_tensor("R")), self.sess.run(w), delta=1E-3)

        psub = px - py
        sub = x - y
        self.assertAlmostEqual(self.sess.run(psub.to_tf_tensor("R")), self.sess.run(sub), delta=1E-3)


if __name__ == '__main__':
    unittest.main()
