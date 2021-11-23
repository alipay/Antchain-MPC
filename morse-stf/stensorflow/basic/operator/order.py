#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : order.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-09 15:49
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
import tensorflow as tf
from typing import Union
from stensorflow.basic.operator.logical import and_op, or_op, nxor_op, not_op
from stensorflow.basic.protocol.compare import greater_SharedTensorBase_SharedTensorBase


def to_bits(x: tf.Tensor, little_endian=True) -> tf.Tensor:
    if x.dtype != tf.int64:
        raise Exception("must have x.dtype==tf.int64")
    y = tf.bitcast(x, 'uint8')
    z = tf.bitwise.bitwise_and(tf.expand_dims(y, axis=-1), [1, 2, 4, 8, 16, 32, 64, 128])
    w = tf.bitwise.right_shift(z, [0, 1, 2, 3, 4, 5, 6, 7])
    new_shape = y.shape.as_list()
    new_shape[-1] = -1
    u = tf.reshape(w, new_shape)
    # if order == "decrease":
    if not little_endian:
        u = tf.reverse(u, axis=[-1])
    return tf.cast(u, 'int64')


def is_negative(x: SharedPairBase) -> SharedPairBase:
    with tf.device(x.ownerL):
        xL_bits = to_bits(x.xL.inner_value, little_endian=False)
        xL_bits_top, xL_bits_low = tf.split(xL_bits, [1, -1], axis=-1)
        xL_bits_top = SharedTensorBase(inner_value=xL_bits_top, module=2)
        xL_bits_low = SharedTensorBase(inner_value=xL_bits_low, module=2)
    with tf.device(x.ownerR):
        xR_bits = to_bits(x.xR.inner_value, little_endian=False)
        xR_bits_top, xR_bits_low = tf.split(xR_bits, [1, -1], axis=-1)
        xR_bits_top = SharedTensorBase(inner_value=xR_bits_top, module=2)
        xR_bits_low = SharedTensorBase(inner_value=xR_bits_low, module=2)

    carry = greater_SharedTensorBase_SharedTensorBase(xL_bits_low, -xR_bits_low + xR_bits_low.ones_like(), x.ownerL,
                                                      x.ownerR)
    x_bits_top = SharedPairBase(xL=xL_bits_top, xR=xR_bits_top, ownerL=x.ownerL, ownerR=x.ownerR, fixedpoint=0)
    x_bits_top = x_bits_top.squeeze(axis=[-1])
    result = carry + x_bits_top
    return result


def is_positive(x: SharedPairBase) -> SharedPairBase:
    return is_negative(x.zeros_like() - x)


def is_not_positive(x: SharedPairBase) -> SharedPairBase:
    y = is_positive(x)
    return y.ones_like() - y


def is_not_negative(x: SharedPairBase) -> SharedPairBase:
    y = is_negative(x)
    return y.ones_like() - y


def less(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase, int, float]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    """

    :param x:
    :param y:
    :return:  1 if x<y else 0
    """
    if isinstance(x, PrivateTensorBase) and isinstance(y, PrivateTensorBase):
        if x.owner == y.owner:
            return PrivateTensorBase.__lt__(x, y)
        else:
            return NotImplemented
    elif isinstance(x, PrivateTensorBase) and isinstance(y, (int, float)):
        inner_value = tf.cast(x.to_tf_tensor(dtype='int64') < y, 'int64')
        return PrivateTensorBase(owner=x.owner, fixedpoint=0, inner_value=inner_value, module=2)
    elif isinstance(x, SharedPairBase) and isinstance(y, (int, float)):
        is_negative_x = is_negative(x)
        if y == 0:
            return is_negative_x
        same_sym_flag = is_negative_x if y < 0 else not_op(is_negative_x)
        same_less = and_op(same_sym_flag, is_negative(x - y))
        diff_less = is_negative_x if y >= 0 else same_sym_flag.zeros_like()
        return or_op(same_less, diff_less)
    elif isinstance(x, SharedPairBase) and isinstance(y, SharedPairBase):
        is_negative_x = is_negative(x)
        is_negative_y = is_negative(y)
        same_sym_flag = nxor_op(is_negative_x, is_negative_y)
        same_less = and_op(same_sym_flag, is_negative(x - y))
        diff_less = and_op(is_negative_x, not_op(is_negative_y))
        return or_op(same_less, diff_less)
    else:
        pass


def leq(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    """

    :param x:
    :param y:
    :return:  1 if x<=y else 0
    """
    y_less_x = less(y, x)
    return not_op(y_less_x)


def greater(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    """

    :param x:
    :param y:
    :return:  1 if x>y else 0
    """
    return less(y, x)


def geq(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    """

    :param x:
    :param y:
    :return:  1 if x>=y else 0
    """
    x_less_y = less(x, y)
    return not_op(x_less_y)
