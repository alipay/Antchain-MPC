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
from stensorflow.basic.operator.truncation import dup_with_precision
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from typing import Union
import warnings
from stensorflow.basic.operator.arithmetic import mul


def softmax_real(x: Union[SharedPair, PrivateTensor]):
    if isinstance(x, PrivateTensor):
        y = tf.nn.softmax(x.to_tf_tensor())
        y = tf.cast(y * (2 ** x.fixedpoint), 'int64')
        z = PrivateTensor(owner=x.owner, fixedpoint=x.fixedpoint,
                          inner_value=y, module=x.module, op_map=x.op_map)
    else:
        y = tf.nn.softmax(x.to_tf_tensor("L"))
        # y = tf.cast(y * (2 ** x.fixedpoint), 'int64')
        # z = PrivateTensor(owner=x.ownerL, fixedpoint=x.fixedpoint,
        #                   inner_value=y, module=x.xL.module, op_map=x.op_map)
        z = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, shape=x.shape, fixedpoint=x.fixedpoint)
        z.load_from_tf_tensor(y)
    return z


def softmax_functionality(x: Union[SharedPair, PrivateTensor]):
    if StfConfig.softmax_iter_num == 0:
        return softmax_real(x)
    elif isinstance(x, PrivateTensor):
        y = x.ones_like() / x.shape[-1]
        x = x / StfConfig.softmax_iter_num
        for _ in range(StfConfig.softmax_iter_num):
            # formula of Qizhi Zhang
            r = (x - (y.expend_dims(axis=[-2]) @ x.expend_dims(axis=-1)).squeeze(axis=-1)) + 1
            y = mul(y, r, fixed_point=y.fixedpoint+r.fixedpoint)
            y = PrivateTensor.from_PrivteTensorBase(y, x.op_map)
            y = y.dup_with_precision(new_fixedpoint=StfConfig.default_fixed_point)
        return y
    elif isinstance(x, SharedPair):
        y = x.to_private("R")
        y = PrivateTensor.from_PrivteTensorBase(y)
        z = softmax_functionality(y)
        return z
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))



def softmax(x: Union[SharedPair, PrivateTensor]):
    if StfConfig.softmax_functionality:
        warnings.warn("Instead of the MPC protocol of softmax, the functionality of softmax is used")
        return softmax_functionality(x)
    if isinstance(x, SharedPair):
        y = x.ones_like() / x.shape[-1]
        x = x / StfConfig.softmax_iter_num
        for _ in range(StfConfig.softmax_iter_num):
            # formula of Qizhi Zhang
            # y = y + (x - (y * x).reduce_sum(axis=-1, keepdims=True)) * y / StfConfig.softmax_iter_num
            # y = y + (x - (y.expend_dims(axis=[-2]) @ x.expend_dims(axis=-1)).squeeze(axis=-1)) * y / StfConfig.softmax_iter_num
            # r = 1 + (x - (y.expend_dims(axis=[-2]) @ x.expend_dims(axis=-1)).squeeze(axis=-1))/ StfConfig.softmax_iter_num
            r = 1 + (x - (y.expend_dims(axis=[-2]) @ x.expend_dims(axis=-1)).squeeze(axis=-1))
            y = mul(y, r, fixed_point=y.fixedpoint+r.fixedpoint)
            y = SharedPair.from_SharedPairBase(y)
            if StfConfig.positive_truncation_without_error:
                y = y.dup_with_precision(new_fixedpoint=StfConfig.default_fixed_point, non_negative=True)
                # y = SharedPair.from_SharedPairBase(y)
            else:
                y = y.dup_with_precision(new_fixedpoint=StfConfig.default_fixed_point)
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

