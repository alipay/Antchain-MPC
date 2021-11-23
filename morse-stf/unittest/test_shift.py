#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_shift
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-12-02 09:45
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import PrivateTensorBase
from stensorflow.basic.protocol.shift import cyclic_lshift
import tensorflow as tf
import numpy as np
import time
import unittest
from stensorflow.engine.start_server import start_local_server
import os

start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()

    def test_cyclic_lshift(self):
        x = PrivateTensorBase(owner="L")
        x.load_from_tf_tensor([[5,6,7],[7,8,9]])  # (np.random.random(size=[8, 4])))
        i = PrivateTensorBase(owner="R", fixedpoint=0, module=3,
                              inner_value=tf.constant([1,2], dtype='int64'))
        z = cyclic_lshift(i, x, prf_flag=True, compress_flag=True)
        self.sess.run(tf.compat.v1.initialize_all_variables())
        z = self.sess.run(z.to_tf_tensor("R"))
        w = [[6,7,5], [9,7,8]]
        self.assertLess(np.linalg.norm(z-w), 1E-3)

