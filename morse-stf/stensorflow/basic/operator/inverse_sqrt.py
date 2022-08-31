#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : inverse_sqrt.py (fake)
   Author : qizhi.zqz
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/5/26 下午2:18
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server
import tensorflow as tf



def inverse_sqrt(x: SharedPair, eps=0.0, y=None):
    # functionality: 1/sqrt(x+eps**2)
    y = x.to_tf_tensor("R")+eps**2
    z = 1/tf.sqrt(y)
    u = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, shape=x.shape)
    u.load_from_tf_tensor(z)
    return u


def careful_inverse_sqrt(x: SharedPair, eps) -> SharedPair:
    # functionality: 1/sqrt(x+eps**2)
    return inverse_sqrt(x, eps)





if __name__ == '__main__':
    start_local_server(config_file="../../../conf/config.json")
    a = [[1.0, 4.0, 1.0, 8.0], [0.0, 16.0, 64.0, 10000.0]]
    # a = [0.0]
    x = SharedPair(ownerL="L", ownerR="R", shape=[2, 4], fixedpoint=16)
    x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
    r = careful_inverse_sqrt(x, 1E-6)
    sess = tf.compat.v1.Session(target=StfConfig.target)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run([r.to_tf_tensor("R")]))
