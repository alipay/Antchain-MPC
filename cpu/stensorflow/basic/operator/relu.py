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


def drelu_binary_log(x: SharedPair, out_carry=False):
    if out_carry:
        t, carry = msb_log_round(x, out_carry)
        s = t.ones_like() - t
        return (s, carry)
    else:
        t = msb_log_round(x)
        s = t.ones_like() - t
    return s


def drelu_binary_linear(x: SharedPair, out_carry=False):
    if out_carry:
        t, carry = msb(x, out_carry)
        s = t.ones_like() - t
        return (s, carry)
    else:
        t = msb(x)
        s = t.ones_like() - t
    # print("s=", s)
    return s


def drelu_binary(x: SharedPair, out_carry=False):
    if StfConfig.drelu == "const":
        if out_carry:
            raise Exception("const drelu unsuported out carry")
        return drelu_binary_const(x)
    elif StfConfig.drelu == "log":
        return drelu_binary_log(x, out_carry)
    elif StfConfig.drelu == "linear":
        return drelu_binary_linear(x, out_carry)
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
    with tf.device(x.owner):
        y = x.to_tf_tensor()
        y = tf.cast(y >= 0, 'int64')
        z = PrivateTensor(owner=x.owner, fixedpoint=0, inner_value=y, module=2)
    return z


def relu(x: Union[SharedPair, PrivateTensor], drelu_b=None):
    if isinstance(x, SharedPair):
        if drelu_b is not None:
            s = drelu_b
        else:
            s = drelu_binary(x)
        y = select_share(s, x)
        y = SharedPair.from_SharedPairBase(y)
        return y
    elif isinstance(x, PrivateTensor):
        if drelu_b is not None:
            s = drelu_b
            y = select_share(s, x)
        else:
            y = relu_local(x)
        return y
    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))


def relu_pull_back(x: Union[SharedPair, PrivateTensor], ploss_py, drelu_b=None):

    if isinstance(ploss_py, SharedPair):
        if drelu_b is not None:
            ploss_px = select_share(drelu_b, ploss_py)
            return SharedPair.from_SharedPairBase(ploss_px)
        else:
            s = drelu_binary(x)
            ploss_px = select_share(s, ploss_py)
            return SharedPair.from_SharedPairBase(ploss_px)
    elif isinstance(ploss_py, PrivateTensor):
        if drelu_b is not None:
            s = drelu_b
        else:
            s = drelu_local(x)
        ploss_px = select_share(s, ploss_py)
        return PrivateTensor.from_PrivteTensorBase(ploss_px)

    else:
        raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))

