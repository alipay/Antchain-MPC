import unittest
from stensorflow.basic.operator.order import is_negative, is_positive, is_not_negative, is_not_positive, less, leq, greater, geq
from stensorflow.basic.basic_class.pair import SharedPair
import tensorflow as tf
from stensorflow.random.random import random_init
import numpy as np
from stensorflow.engine.start_server import start_local_server

import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()
    def test_is_negative(self):
        a=np.random.uniform(low=-1.0, high=1.0, size=[10,5])
        x=SharedPair(ownerL="L", ownerR="R", shape=[2])
        x.load_from_tf_tensor(tf.constant(a))
        y=is_negative(x)
        y=y.to_tf_tensor("R")
        random_init(self.sess)
        result=self.sess.run(y)
        self.assertTrue(np.array_equal(result,(a<0)))
        
        

    def test_is_positive(self):
        a=np.random.uniform(low=-1.0, high=1.0, size=[10,5])
        
        x=SharedPair(ownerL="L", ownerR="R", shape=[2])
        x.load_from_tf_tensor(tf.constant(a))
        y=is_positive(x)
        y=y.to_tf_tensor("R")
        random_init(self.sess)
        result=self.sess.run(y)
        self.assertTrue(np.array_equal(result, (a>0)))

    def test_is_not_positive(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=[10, 5])
        x = SharedPair(ownerL="L", ownerR="R", shape=[2])
        x.load_from_tf_tensor(tf.constant(a))
        y = is_not_positive(x)
        y = y.to_tf_tensor("R")
        random_init(self.sess)
        result=self.sess.run(y)
        self.assertTrue(np.array_equal(result,(a<=0)))
        
        

    def test_is_not_negative(self):
        a=np.random.uniform(low=-1.0, high=1.0, size=[10,5])
        
        x=SharedPair(ownerL="L", ownerR="R", shape=[2])
        x.load_from_tf_tensor(tf.constant(a))
        y=is_not_negative(x)
        y=y.to_tf_tensor("R")
        random_init(self.sess)
        result=self.sess.run(y)
        self.assertTrue(np.array_equal(result,(a>=0)))
        

    def test_less(self):
        a=np.random.uniform(low=-1.0, high=1.0, size=[10,5])
        x=SharedPair(ownerL="L", ownerR="R", shape=[2])
        x.load_from_tf_tensor(tf.constant(a))
        b=np.random.uniform(low=-1.0, high=1.0, size=[10,5])
        y=SharedPair(ownerL="L", ownerR="R", shape=[2])
        y.load_from_tf_tensor(tf.constant(b))
        c=(a<b)
        z=less(x, y)
        random_init(self.sess)
        result = self.sess.run(z.to_tf_tensor("R"))
        self.assertTrue(np.array_equal(result, c))


if __name__ == '__main__':
    unittest.main()
