#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""

import tensorflow as tf
from tensorflow.python.util.compat import collections_abc
from stensorflow.basic.basic_class.base import SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateTensorBase
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair, SharedPairBase
from stensorflow.basic.operator.relu import drelu_binary
from stensorflow.basic.operator.selectshare import select_share
from typing import Union
from stensorflow.global_var import StfConfig
from stensorflow.basic.operator.sigmoid import sin2pi
import numpy as np
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
        raise TypeError(
            "Expected PrivateTensorBase or SharedPair  for 'input' argument to 'average' Op, not %r." % input)


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
        raise TypeError(
            "Expected PrivateTensorBase or SharedPairBase for 'out_backprop' argument to 'average' Op, not %r." % out_backprop)


def max_pool_div2(x: SharedPair,
                  axis, return_index=False):
    """
    :param x:  A SharedPair of shape [d1,d2,...,di,...dn]
    :param axis:  the i
    :param return_index:  bool
    :return:  if return_index=False
                 A SharedPair of shape [d1,d2,...,di/2,...dn]
            else:
                a tuple of A SharedPair of shape [d1,d2,...,di/2,...dn] and the
                    SharedPair of shape  [d1,d2,...,di/2,...dn] of module==2
    """
    xshape = x.shape
    xshape = xshape[0:axis] + [xshape[axis] // 2, 2] + xshape[axis + 1:]
    # print("xshape=", xshape)
    x = x.reshape(xshape)
    x0, x1 = x.split(size_splits=[1, 1], axis=axis + 1)
    x0 = x0.squeeze(axis=axis + 1)
    x1 = x1.squeeze(axis=axis + 1)
    # print("x0=", x0)
    # print("x1=", x1)
    s = drelu_binary(x1-x0)
    # print("s=", s)
    y = select_share(s, x1, x0)
    if return_index:
        return y, s
    else:
        return y


def max_pool_div2_back(ploss_py: SharedPair, axis, s: SharedPair):
    """
    (ploss_py,0) if s==0
    (0, ploss_py) if s==1

    :param ploss_py: SharedPair of shape [d1,...,di,...dn]
    :param axis:  i
    :param s:  SharedPair of shape [d1,...,di,...dn]
    :return:  SharedPair of shape [d1,...,2di,...dn]
    """
    if s.xL.module != 2:
        raise Exception("must have s.module==2")
    if ploss_py.shape[:axis] != s.shape[:axis]:
        raise Exception("must have ploss_py.shape[:axis]==s.shape[:axis]")
    if ploss_py.shape[axis + 1:] != s.shape[axis + 1:]:
        raise Exception("must have ploss_py.shape[axis+1:]==s.shape[axis+1:], "
                        "but ploss_py.shape={}, s.shape={}, axis+1={}".format(ploss_py.shape, s.shape, axis+1))
    # print("s=", s)
    # print("ploss_py=", ploss_py)
    new_shape = ploss_py.shape[:axis] + [-1] + ploss_py.shape[axis + 1:]
    ploss_py = ploss_py.expend_dims(axis=axis+1)
    s = s.expend_dims(axis=axis+1)
    sploss_py = select_share(s, ploss_py)
    ploss_px = (ploss_py - sploss_py).concat(sploss_py, axis=axis+1)
    ploss_px = ploss_px.reshape(new_shape)
    return ploss_px


def max_pool2d(x: Union[PrivateTensor, SharedPair],
               ksize, return_s=False):
    if len(ksize) == 2:
        ksize = [1, ksize[0], ksize[1], 1]
    if len(x.shape) != len(ksize):
        raise Exception("must have len(x.shape) == len(ksize), but x.shape={}, ksize={}".format(x.shape, ksize))
    if return_s:
        index_list = []
        for axis in range(len(ksize)):
            size_in_axis = ksize[axis]
            while size_in_axis > 1:
                x, index = max_pool_div2(x, axis, return_index=True)
                size_in_axis //= 2
                index_list.append(index)
        return x, index_list
    else:
        for axis in range(len(ksize)):
            size_in_axis = ksize[axis]
            while size_in_axis > 1:
                x = max_pool_div2(x, axis)
                size_in_axis //= 2
        return x


def max_pool2d_back(ploss_py: SharedPair, ksize, index_list=None, x: Union[PrivateTensor, SharedPair] = None):
    # print("x=", x)
    if len(ksize)==2:
        ksize = [1, ksize[0], ksize[1], 1]
    #print("ksize=", ksize)
    #print("ploss_py=", ploss_py)
    #print("index_list=", index_list)
    if index_list is None:
        y, index_list = max_pool2d(x, ksize, return_s=True)
    i = len(index_list) - 1
    ploss_px = ploss_py
    for axis in range(len(ksize) - 1, -1, -1):
        # print("axis=", axis)
        size_in_axis = ksize[axis]
        # print("size_in_axis=", size_in_axis)
        while size_in_axis > 1:
            ploss_px = max_pool_div2_back(ploss_px, axis=axis, s=index_list[i])
            i -= 1
            size_in_axis //= 2
    return ploss_px
