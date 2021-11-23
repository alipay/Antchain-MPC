#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""

import tensorflow as tf
from tensorflow.python.util.compat import collections_abc
from stensorflow.basic.basic_class.base import SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateTensorBase
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair,SharedPairBase
from typing import Union
from stensorflow.global_var import StfConfig


PModel = StfConfig.pool_module


def get_sequence(value, n, channel_index, name):
    """
    Formats a value input for avg_pool2d.
    :param value:
    :param n:
    :param channel_index:
    :param name:
    :return:
    """
    if value is None:
        value = [1]
    elif not isinstance(value, collections_abc.Sized):
        value = [value]

    current_n = len(value)
    if current_n == n + 2:
        return value
    elif current_n == 1:
        value = list((value[0],) * n)
    elif current_n == n:
        value = list(value)
    else:
        raise ValueError("{} should be of length 1, {} or {} but was {}".format(
            name, n, n + 2, current_n))

    if channel_index == 1:
        return [1, 1] + value
    else:
        return [1] + value + [1]


def avg_pool2d(input: Union[PrivateTensor, SharedPair],
               ksize,
               strides,
               padding,
               data_format="NHWC"):
    """
    See tf.avg_pool2d
      """
    if data_format is None:
        data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3
    ksize = get_sequence(ksize, 2, channel_index, "ksize")
    strides = get_sequence(strides, 2, channel_index, "strides")
    if isinstance(input, PrivateTensorBase):
        with tf.device(input.owner):
            v = PModel.int64_avg_pool(input.inner_value, ksize=ksize, strides=strides, padding=padding)
            z = PrivateTensor(owner=input.owner, inner_value=v, fixedpoint=input.fixedpoint)
        return z
    elif isinstance(input, SharedPairBase):
        with tf.device(input.ownerL):
            vL = PModel.int64_avg_pool(input.xL.inner_value, ksize=ksize, strides=strides, padding=padding)
        with tf.device(input.ownerR):
            vR = PModel.int64_avg_pool(input.xR.inner_value, ksize=ksize, strides=strides, padding=padding)
        z = SharedPair(ownerL=input.ownerL,
                       ownerR=input.ownerR,
                       xL=SharedTensorBase(inner_value=vL, module=input.xL.module, shape=list(vL.shape)),
                       xR=SharedTensorBase(inner_value=vR, module=input.xR.module, shape=list(vR.shape)),
                       fixedpoint=input.fixedpoint)
        return z

    else:
        raise TypeError("Expected PrivateTensorBase or SharedPair  for 'input' argument to 'average' Op, not %r." % input)


def sum_pool2d_grad(input_shape,
                    out_backprop: Union[SharedPair, PrivateTensor],
                    ksize,
                    strides,
                    padding,
                    data_format="NHWC"):
    """
    The operation to compute SumPool gradients.
    :param input_shape: See tf.avg_pool2d_grad
    :param out_backprop: See tf.avg_pool2d_grad
    :param ksize: See tf.avg_pool2d_grad
    :param strides: See tf.avg_pool2d_grad
    :param padding: See tf.avg_pool2d_grad
    :param data_format: See tf.avg_pool2d_grad
    :return:
    """
    if data_format is None:
        data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3
    ksize = get_sequence(ksize, 2, channel_index, "ksize")
    strides = get_sequence(strides, 2, channel_index, "strides")
    if isinstance(out_backprop, PrivateTensorBase):
        with tf.device(out_backprop.owner):
            g = PModel.sum_pool_grad(input_shape, grad=out_backprop.inner_value,
                                           ksize=ksize, strides=strides, padding=padding)
            z = PrivateTensor(owner=out_backprop.owner, fixedpoint=out_backprop.fixedpoint, inner_value=g)
            return z
    elif isinstance(out_backprop, SharedPairBase):
        with tf.device(out_backprop.ownerL):
            gL = PModel.sum_pool_grad(input_shape, grad=out_backprop.xL.inner_value,
                                 ksize=ksize, strides=strides, padding=padding)
        with tf.device(out_backprop.ownerR):
            gR = PModel.sum_pool_grad(input_shape, grad=out_backprop.xR.inner_value,
                                 ksize=ksize, strides=strides, padding=padding)
        z = SharedPair(ownerL=out_backprop.ownerL,
                       ownerR=out_backprop.ownerR,
                       xL=SharedTensorBase(inner_value=gL, module=out_backprop.xL.module, shape=list(gL.shape)),
                       xR=SharedTensorBase(inner_value=gR, module=out_backprop.xR.module, shape=list(gR.shape)),
                       fixedpoint=out_backprop.fixedpoint)
        return z
    else:
        raise TypeError("Expected PrivateTensorBase or SharedPairBase for 'out_backprop' argument to 'average' Op, not %r." % out_backprop)










