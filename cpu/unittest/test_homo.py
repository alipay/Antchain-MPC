#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_homo
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/9/29 下午5:43
   Description : description what the main function of this file
"""

from stensorflow.engine.start_server import start_local_server, start_client
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.homo_enc.homo_enc import homo_init
from stensorflow.basic.basic_class.private import PrivateTensor
import numpy as np
import unittest
start_local_server(config_file="./conf/config_parties2.json")
class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")

    def tearDown(self):
        self.sess.close()



    def test_homo_mul(self):
        x = PrivateTensor(owner="L")
        a = np.random.random([300,14])
        x.load_from_numpy(a)

        y = PrivateTensor(owner="R")
        b = np.random.random([300,14])
        y.load_from_numpy(b)

        z = x*y

        sess = tf.compat.v1.Session(StfConfig.target)
        homo_init(sess)
        self.assertLessEqual(np.sum(np.abs(sess.run(z.to_tf_tensor("R"))-a*b)), 1E-6)

    def test_homo_matmul(self):
        x = PrivateTensor(owner="L")
        a = np.random.random([300,14])
        x.load_from_numpy(a)

        y = PrivateTensor(owner="R")
        b = np.random.random([14,11])
        y.load_from_numpy(b)

        z = x@y

        sess = tf.compat.v1.Session(StfConfig.target)
        homo_init(sess)
        self.assertLessEqual(np.sum(np.abs(sess.run(z.to_tf_tensor("R"))-a@b)), 1E-6)



