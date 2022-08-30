#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : relu
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-10 15:22
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.operator.selectshare import select_share
from typing import Union
from stensorflow.global_var import StfConfig
from stensorflow.exception.exception import StfTypeException, StfValueException
from stensorflow.basic.protocol.msb import msb, msb_log_round
from stensorflow.basic.operator.order import is_positive
import tensorflow as tf


def drelu_binary_const(x: SharedPair):
    y = is_positive(x)
    return SharedPair.from_SharedPairBase(y)


def drelu_binary_log(x: SharedPair):
    t = msb_log_round(x)
    return t.ones_like() - t


def drelu_binary_linear(x: SharedPair):
    t = msb(x)
    # print("t=", t)
    s = t.ones_like() - t
    # print("s=", s)
    return s


def drelu_binary(x: SharedPair):
    if StfConfig.drelu == "const":
        return drelu_binary_const(x)
    elif StfConfig.drelu == "log":
        return drelu_binary_log(x)
    elif StfConfig.drelu == "linear":
        return drelu_binary_linear(x)
    else:
        raise StfValueException("StfConfig.delay", "high or middle or low", StfConfig.delay)


def relu_local(x: PrivateTensor):
    z = PrivateTensor(owner=x.owner)
    with tf.device(x.owner):
        x = x.to_tf_tensor()
        y = tf.nn.relu(x)
        z.load_from_tf_tensor(y)
    return z


def drelu_local(x: PrivateTensor):
    z = PrivateTensor(owner=x.owner)
    with tf.device(x.owner):
        x = x.to_tf_tensor()
        y = tf.cast(x >= 0, 'float32')
        z.load_from_tf_tensor(y)
    return z


def relu(x: Union[SharedPair, PrivateTensor], drelu_b=None):
    if isinstance(x, SharedPair):
        if drelu_b is not None:
            s = drelu_b
        else:
            s = drelu_binary(x)
        y = select_share(s, x)
        return SharedPair.from_SharedPairBase(y)
    elif isinstance(x, PrivateTensor):
        return relu_local(x)
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))


def relu_pull_back(x: Union[SharedPair, PrivateTensor], ploss_py, drelu_b=None):
    if drelu_b is not None:
        ploss_px = select_share(drelu_b, ploss_py)
        return SharedPair.from_SharedPairBase(ploss_px)
    elif isinstance(x, SharedPair):
        s = drelu_binary(x)
        ploss_px = select_share(s, ploss_py)
        return SharedPair.from_SharedPairBase(ploss_px)
    elif isinstance(x, PrivateTensor):
        return drelu_local(x) * ploss_py
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))

