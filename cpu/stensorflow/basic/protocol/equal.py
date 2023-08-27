#!/usr/bin/env python
# coding=utf-8
"""
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : iszero.py
   Author : Qizhi Zhang
   Email: zqz.math@gmail.ccom
   Create Time : 2022/12/25 上午11:57
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import SharedPairBase, SharedTensorBase
from stensorflow.basic.basic_class.bitwise import PrivateTensorBitwise, SharedTensorBitwise
import tensorflow as tf
from typing import Tuple, Union
import numpy as np
import math
from stensorflow.exception.exception import StfDTypeException


def to_bool(x: tf.Tensor) -> tf.Tensor:
    """

    :param x:  tf.Tensor  shape =[s1,..., sn]
    :return:   tf.Tensor  shape =[s1,..., sn, sizeof(x.dtype)]
    """
    y = tf.bitcast(x, 'uint8')
    z = tf.bitwise.bitwise_and(tf.expand_dims(y, axis=-1), [1, 2, 4, 8, 16, 32, 64, 128])
    w = tf.bitwise.right_shift(z, [0, 1, 2, 3, 4, 5, 6, 7])
    new_shape = y.shape.as_list()
    new_shape[-1] = -1
    u = tf.reshape(w, new_shape)
    z = tf.cast(u, 'bool')
    return z


def bool_to_int64(z: tf.Tensor) -> tf.Tensor:
    """

    :param z: tf.Tensor of dtype bool  z.shape%64==0
    :return:  tf.Tensor of dtype int64 and shape = z.shape/64
    """
    u = tf.cast(z, 'uint8')
    u = tf.reshape(u, [-1, 8])
    u *= [1, 2, 4, 8, 16, 32, 64, 128]

    u = tf.reduce_sum(u, axis=[-1])

    u = tf.reshape(u, [-1, 8])
    u = tf.bitcast(u, 'int64')

    new_shape = z.shape.as_list()
    if new_shape[-1] % 64 != 0:
        raise Exception("must have new_shape[-1]%64 == 0")
    new_shape[-1] //= 64
    if new_shape[-1] == 1:
        new_shape = new_shape[0:-1]

    u = tf.reshape(u, new_shape)

    return u


def pad_to_64n(x: tf.Tensor):
    x = tf.reshape(x, [-1])
    if x.shape[0] % 64 == 0:
        return x
    else:
        n = math.ceil(x.shape[0] / 64)
        x = tf.pad(x, paddings=[[0, 64 * n - x.shape[0]]])
        return x


def bool_expansion(x: tf.Tensor):
    """

    :param x: tf.Tensor
    :return:  List of tf.Tensor of same type of x of shape [ceil([-1]/64)]
               bool expansion keeping the order
    """
    # print("x@line82=",x)
    if x.dtype in [tf.int64, tf.uint64, tf.float64, tf.int32, tf.uint32, tf.float32, tf.int16, tf.uint16, tf.float16]:
        bit_size = x.dtype.size * 8
    else:
        raise StfDTypeException("x", "tf.intx,  tf.uintx, tf.floatx for x in 16,32,64", x.dtype)

    z = to_bool(x)
    list_z = tf.split(z, num_or_size_splits=bit_size, axis=-1)
    if x.dtype in [tf.float16, tf.float32, tf.float64]:
        for i in range(bit_size - 1):
            list_z[i] = tf.math.logical_xor(list_z[i], list_z[-1])
        list_z[-1] = tf.math.logical_not(list_z[-1])
    list_z = list(map(lambda zi: bool_to_int64(pad_to_64n(zi)), list_z))
    # print("list_z[0]=", list_z[0])
    return list_z

def equal(xL: SharedTensorBase, xR: SharedTensorBase, ownerL, ownerR) -> SharedPairBase:
    if xL.shape != xR.shape:
        raise Exception("must have xL.shape==xR.shape")
    if xL.module is not None:
        if (xL.module != xR.module):
            raise Exception("must have xL.module==xR.module")
        bit_length = int(math.ceil(math.log2(xL.module)))
    else:
        bit_length = 64
    with tf.device(ownerL):
        list_xL = bool_expansion(xL.inner_value)
        list_xL = list_xL[0:bit_length]
        list_xL = list(
            map(lambda xi: PrivateTensorBitwise(owner=ownerL, inner_value=SharedTensorBitwise(inner_value=xi)),
                list_xL))

    with tf.device(ownerR):
        list_xR = bool_expansion(xR.inner_value)
        list_xR = list_xR[0:bit_length]
        list_xR = list(
            map(lambda xi: PrivateTensorBitwise(owner=ownerR, inner_value=SharedTensorBitwise(inner_value=xi)),
                list_xR))
    bit_equal = ~(list_xL[0] + list_xR[0])
    for i in range(1, len(list_xL)):
        bit_equal = bit_equal * ~(list_xL[i] + list_xR[i])
    bit_equal = bit_equal.toSharedPairBase(shape=xL.shape)
    return bit_equal





def special_mul(x, y):
    """
    :param x: =(x0, x1)
    :param y: =(y0, y1)
    :return: (x0y0, x1y2+x2)
    """
    return x[0] * y[0], x[0] * y[1] + x[1]


def reduce_special_mul(list_mat):
    """

    :param list_mat:
    :return:
    """
    lengh = len(list_mat)
    if lengh == 1:
        return list_mat[0]
    else:
        list0 = list_mat[0: lengh // 2]
        list1 = list_mat[lengh // 2: lengh]
        r0 = reduce_special_mul(list0)
        r1 = reduce_special_mul(list1)
        return special_mul(r0, r1)

#
# def msb_log_round(x: SharedPair, out_carry=False) -> Union[SharedPair, Tuple[SharedPair, SharedPair]]:
#     with tf.device(x.ownerL):
#         list_xL = bool_expansion(x.xL.inner_value)
#         list_xL = list(
#             map(lambda xi: PrivateTensorBitwise(owner=x.ownerL, inner_value=SharedTensorBitwise(inner_value=xi)),
#                 list_xL))
#
#     with tf.device(x.ownerR):
#         list_xR = bool_expansion(x.xR.inner_value)
#         list_xR = list(
#             map(lambda xi: PrivateTensorBitwise(owner=x.ownerR, inner_value=SharedTensorBitwise(inner_value=xi)),
#                 list_xR))
#
#     matrix_list = [(xL + xR, xL * xR) for (xL, xR) in zip(list_xL, list_xR)]
#     matrix_list.reverse()
#     carry = reduce_special_mul(matrix_list[1:])[1]
#     msb = list_xL[-1] + list_xR[-1] + carry
#     if out_carry:
#         carry = (list_xL[-1] + carry) * (list_xR[-1] + carry) - carry
#     msb = msb.toSharedPairBase(shape=x.shape)
#     if out_carry:
#         carry = carry.toSharedPairBase(shape=x.shape)
#         return msb, carry
#     else:
#         return msb
#
#
