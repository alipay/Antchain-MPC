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
from stensorflow.basic.basic_class.private import PrivateTensor
from typing import Union


def softmax(x: Union[SharedPair, PrivateTensor]):
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
