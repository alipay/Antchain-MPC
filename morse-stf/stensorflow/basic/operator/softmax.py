#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : softmax
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-12-01 14:17
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.pair import SharedPair
import tensorflow as tf
from stensorflow.exception.exception import StfTypeException
from stensorflow.basic.operator.relu import relu
from stensorflow.basic.operator.sigmoid import sin2pi
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from typing import Union


def softmax_bak(x: Union[SharedPair, PrivateTensor]):
    if isinstance(x, SharedPair):
        y = relu(x) # + x.ones_like()
        z = y.reduce_sum(axis=1, keepdims=True)
        # return ~ (sin2pi(z, T=2**41)*(2**40/np.pi)) * y
        return ~sin2pi(z, T=2 ** 14) * y / (2 ** 13 / np.pi)
    elif isinstance(x, PrivateTensor):
        with tf.device(x.owner):
            y = tf.nn.softmax(x.to_tf_tensor())
            y = tf.cast(y * (2 ** x.fixedpoint), 'int64')
            z = PrivateTensor(owner=x.owner, fixedpoint=x.fixedpoint,
                              inner_value=y, module=x.module, op_map=x.op_map)
            return z
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))



def softmax(x: Union[SharedPair, PrivateTensor]):
    if isinstance(x, SharedPair):
        y = x.ones_like() / x.shape[-1]
        for _ in range(StfConfig.softmax_iter_num):
            # formula of Qizhi Zhang
            # y = y + (x - (y * x).reduce_sum(axis=-1, keepdims=True)) * y / StfConfig.softmax_iter_num
            y = y + (x - (y.expend_dims(axis=[-2]) @ x.expend_dims(axis=-1)).squeeze(axis=-1)) * y / StfConfig.softmax_iter_num

        return y
    elif isinstance(x, PrivateTensor):
        with tf.device(x.owner):
            y = tf.nn.softmax(x.to_tf_tensor())
            y = tf.cast(y * (2 ** x.fixedpoint), 'int64')
            z = PrivateTensor(owner=x.owner, fixedpoint=x.fixedpoint,
                              inner_value=y, module=x.module, op_map=x.op_map)
            return z
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))
