#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : domain_determin
   Author : qizhi.zqz
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/5/26 下午2:18
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.pair import SharedPair, SharedTensorBase
from stensorflow.basic.basic_class.bitwise import PrivateTensorBitwise, SharedTensorBitwise, SharedPairBitwise
from stensorflow.basic.protocol.msb import bool_expansion, to_bool
from stensorflow.basic.operator.selectshare import select_share
import numpy as np
from typing import List
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client
import tensorflow as tf
import random
import time
import math

random.seed(0)
"""
A Example of training a LR model on a dataset of feature number 291 and predict using 
this model.
The features are in the party L, the label is in the party R.
"""


def functionality_invers_sqrt(x: SharedPair, eps=0.0):
    # y = 1 / (tf.sqrt(x.to_tf_tensor("R")) + eps)
    y = 1 / (tf.sqrt(x.to_tf_tensor("R") + eps * eps))
    z = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, shape=y.shape)
    z.load_from_tf_tensor(y)
    return z


def A2bitwisePair(x: SharedPair) -> List[SharedPairBitwise]:
    """

    :param x:  [a1,...,an]
    :return:   list of [a1x...xan/64] len=64
    """
    x = x.reshape([-1])
    with tf.device(x.ownerL):
        list_xL = bool_expansion(x.xL.inner_value)
        list_xL = list(
            map(lambda xi: PrivateTensorBitwise(owner=x.ownerL, inner_value=SharedTensorBitwise(inner_value=xi)),
                list_xL))

    with tf.device(x.ownerR):
        list_xR = bool_expansion(x.xR.inner_value)
        list_xR = list(
            map(lambda xi: PrivateTensorBitwise(owner=x.ownerR, inner_value=SharedTensorBitwise(inner_value=xi)),
                list_xR))

    result_list = [list_xL[0] + list_xR[0]]
    carry = list_xL[0] * list_xR[0]
    for i in range(1, len(list_xL) - 1):
        result_list.append(list_xL[i] + list_xR[i] + carry)
        carry = (list_xL[i] + carry) * (list_xR[i] + carry) - carry

    msb = list_xL[-1] + list_xR[-1] + carry
    result_list.append(msb)
    return result_list


def invert_one_hot(lx: List[SharedPairBitwise], fixedpoint=StfConfig.default_fixed_point, rshift=0) -> List[SharedPairBitwise]:
    """

    :param lx:  one-hot
    :param fixedpoint:
    param rshift:  i
    :return: 1/2^ix
    """
    invert_lx = []
    if rshift>0:
        shifted_lx = [lx[0].zeros_like()]
        for i in range(len(lx)-1):
            shifted_lx.append(lx[i])
        lx = shifted_lx
    for i in range(len(lx)):
        reflect_i = 2 * fixedpoint - i
        if 0 <= reflect_i < len(lx):
            invert_lx += [lx[reflect_i]]
        else:
            invert_lx += [lx[0].zeros_like()]
    return invert_lx


def one_hot_bitwisePair2A(x: List[SharedPairBitwise], shape, fixedpoint) -> SharedPair:
    if fixedpoint % 2 != 0:
        raise Exception("must have fixedpoint%2==0")
    with tf.device(x[0].ownerL):
        xL = to_bool(x[0].xL.inner_value)
        xL = tf.cast(xL, 'int64')
        even_xL = SharedTensorBase(inner_value=xL)
    with tf.device(x[0].ownerR):
        xR = to_bool(x[0].xR.inner_value)
        xR = tf.cast(xR, 'int64')
        even_xR = SharedTensorBase(inner_value=xR)

    with tf.device(x[1].ownerL):
        xL = to_bool(x[1].xL.inner_value)
        xL = tf.cast(xL, 'int64')
        odd_xL = SharedTensorBase(inner_value=xL)
    with tf.device(x[1].ownerR):
        xR = to_bool(x[1].xR.inner_value)
        xR = tf.cast(xR, 'int64')
        odd_xR = SharedTensorBase(inner_value=xR)

    for i in range(1, len(x)):
        with tf.device(x[i].ownerL):
            xL = to_bool(x[i].xL.inner_value)
            xL = tf.cast(xL, 'int64')
            if i % 2 == 0:
                even_xL += SharedTensorBase(inner_value=xL * (1 << (i // 2)))
            else:
                odd_xL += SharedTensorBase(inner_value=xL * (1 << (i // 2)))
        with tf.device(x[i].ownerR):
            xR = to_bool(x[i].xR.inner_value)
            xR = tf.cast(xR, 'int64')
            if i % 2 == 0:
                even_xR += SharedTensorBase(inner_value=xR * (1 << (i // 2)))
            else:
                odd_xR += SharedTensorBase(inner_value=xR * (1 << (i // 2)))
    y_even = SharedPair(ownerL=x[0].ownerL, ownerR=x[0].ownerR, xL=even_xL, xR=-even_xR, fixedpoint=fixedpoint // 2)
    y_odd = SharedPair(ownerL=x[0].ownerL, ownerR=x[0].ownerR, xL=odd_xL, xR=-odd_xR, fixedpoint=fixedpoint // 2)
    y = (y_even + np.sqrt(2) * y_odd) ** 2
    y = y.reshape([-1])
    print("y@line 118=", y)
    print("shape@line120=", shape)
    if y.shape[0] != np.prod(shape):
        y = y[0: np.prod(shape)]
    return y.reshape(shape)


def one_hot_bitwisePair_sqrt2A(x: List[SharedPairBitwise], fixedpoint, shape) -> SharedPair:
    if (fixedpoint % 4 != 0):
        raise Exception("must have fixedpoint%4 == 0")
    list_xL = [0, 0, 0, 0]
    list_xR = [0, 0, 0, 0]
    z = 0
    for i in range(len(x)):
        with tf.device(x[i].ownerL):
            xL = to_bool(x[i].xL.inner_value)
            xL = tf.cast(xL, 'int64')
            if i // 4 == 0:
                list_xL[i % 4] = SharedTensorBase(inner_value=xL * (1 << (i // 4)))
            else:
                list_xL[i % 4] += SharedTensorBase(inner_value=xL * (1 << (i // 4)))
        with tf.device(x[i].ownerR):
            xR = to_bool(x[i].xR.inner_value)
            xR = tf.cast(xR, 'int64')
            if i // 4 == 0:
                list_xR[i % 4] = SharedTensorBase(inner_value=xR * (1 << (i // 4)))
            else:
                list_xR[i % 4] += SharedTensorBase(inner_value=xR * (1 << (i // 4)))

    for j in range(4):
        y = SharedPair(ownerL=x[0].ownerL, ownerR=x[0].ownerR, xL=list_xL[j], xR=-list_xR[j],
                       fixedpoint=fixedpoint // 4)
        z += np.power(2, j / 4) * y
    w = z ** 2
    w = w.reshape([-1])
    if w.shape[0] != np.prod(shape):
        w = w[0: np.prod(shape)]
    return w.reshape(shape)


def inverse_sqrt(x: SharedPair, eps=0.0, y=None):
    # 1/sqrt(x+eps**2)
    if isinstance(x, SharedPair):
        if y is None:
            y = x.ones_like() / np.sqrt(2)
        for _ in range(StfConfig.inv_sqrt_iter_num):
            if eps == 0.0:
                y = 1.5 * y - 0.5 * x * y ** 3
            else:
                y = 1.5 * y - 0.5 * x * y ** 3 - 0.5 * (eps * y) ** 2 * y
        return y
    else:
        Exception(NotImplementedError)


def careful_inverse_sqrt(x: SharedPair, eps) -> SharedPair:
    # 1/sqrt(x+eps**2)
    # -------------- find a closed to x and is 2-power ---------------------
    # careful_fixedpoint = -(math.floor(math.log2(eps))//2*4)
    # print("careful_fixedpoint=", careful_fixedpoint)
    careful_fixedpoint = 16
    _x = x.dup_with_precision(new_fixedpoint=careful_fixedpoint) + eps * eps
    # print("_x=", _x)
    x_bitwise = A2bitwisePair(_x)
    accum_x_bitwise = [None] * (len(x_bitwise) - 1) + [x_bitwise[-1]]  # accum
    for i in range(len(x_bitwise) - 2, -1, -1):
        accum_x_bitwise[i] = accum_x_bitwise[i + 1] + x_bitwise[i] + accum_x_bitwise[i + 1] * x_bitwise[i]

    is_not_zero_flag = accum_x_bitwise[0]
    # print("is_not_zero_flag=", is_not_zero_flag)
    assert isinstance(is_not_zero_flag, SharedPairBitwise)

    with tf.device(is_not_zero_flag.ownerL):
        is_not_zeroL = to_bool(is_not_zero_flag.xL.inner_value)
        is_not_zeroL = tf.cast(is_not_zeroL, 'int64')
        is_not_zeroL = SharedTensorBase(inner_value=is_not_zeroL, module=2)
    with tf.device(is_not_zero_flag.ownerR):
        is_not_zeroR = to_bool(is_not_zero_flag.xR.inner_value)
        is_not_zeroR = tf.cast(is_not_zeroR, 'int64')
        is_not_zeroR = SharedTensorBase(inner_value=is_not_zeroR, module=2)
    is_not_zero_flag = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=is_not_zeroL,
                                  xR=is_not_zeroR, fixedpoint=0)
    is_not_zero_flag = is_not_zero_flag.reshape([-1])
    # print("is_not_zero_flag=", is_not_zero_flag)
    if is_not_zero_flag.shape[0] != np.prod(x.shape):
        is_not_zero_flag = is_not_zero_flag[0: np.prod(x.shape)]
    is_not_zero_flag = is_not_zero_flag.reshape(x.shape)

    diff_accumor_x_bitwise = [None] * (len(x_bitwise) - 1) + [accum_x_bitwise[-1]]
    for i in range(len(x_bitwise) - 2, -1, -1):
        diff_accumor_x_bitwise[i] = accum_x_bitwise[i] - accum_x_bitwise[i + 1]
    a = diff_accumor_x_bitwise
    # print("a[0]@line 197=", a[0])
    # -----------------------v = a^{-1}---------------------------------------
    z = invert_one_hot(a, fixedpoint=careful_fixedpoint)
    # print("z[0]@line 200=", z[0])
    v = one_hot_bitwisePair2A(z, shape=x.shape, fixedpoint=careful_fixedpoint)
    # print("v@line 202=", v)

    # --------------- u = a^{-1/2} ---------------------------

    u = one_hot_bitwisePair_sqrt2A(z, fixedpoint=careful_fixedpoint, shape=x.shape)
    # u = u.dup_with_precision(x.fixedpoint)
    # --------------------------------------------------
    w = inverse_sqrt(_x * v) * u
    w = w.dup_with_precision(x.fixedpoint)
    w = select_share(is_not_zero_flag, w, 1 / eps * w.ones_like())
    return w





def careful_inverse_sqrt2(x: SharedPair, eps) -> SharedPair:
    # may lead to overflow
    # 1/sqrt(x+eps**2)
    # -------------- find a closed to x and is 2-power ---------------------
    # careful_fixedpoint = -(math.floor(math.log2(eps))//2*4)
    # print("careful_fixedpoint=", careful_fixedpoint)
    careful_fixedpoint = 16
    _x = x.dup_with_precision(new_fixedpoint=careful_fixedpoint) + eps * eps
    print("_x=", _x)
    x_bitwise = A2bitwisePair(_x)
    accum_x_bitwise = [None] * (len(x_bitwise) - 1) + [x_bitwise[-1]]  # accum
    for i in range(len(x_bitwise) - 2, -1, -1):
        accum_x_bitwise[i] = accum_x_bitwise[i + 1] + x_bitwise[i] + accum_x_bitwise[i + 1] * x_bitwise[i]

    is_not_zero_flag = accum_x_bitwise[0]
    print("is_not_zero_flag=", is_not_zero_flag)
    assert isinstance(is_not_zero_flag, SharedPairBitwise)

    with tf.device(is_not_zero_flag.ownerL):
        is_not_zeroL = to_bool(is_not_zero_flag.xL.inner_value)
        is_not_zeroL = tf.cast(is_not_zeroL, 'int64')
        is_not_zeroL = SharedTensorBase(inner_value=is_not_zeroL, module=2)
    with tf.device(is_not_zero_flag.ownerR):
        is_not_zeroR = to_bool(is_not_zero_flag.xR.inner_value)
        is_not_zeroR = tf.cast(is_not_zeroR, 'int64')
        is_not_zeroR = SharedTensorBase(inner_value=is_not_zeroR, module=2)
    is_not_zero_flag = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=is_not_zeroL,
                                  xR=is_not_zeroR, fixedpoint=0)
    is_not_zero_flag = is_not_zero_flag.reshape([-1])
    print("is_not_zero_flag=", is_not_zero_flag)
    if is_not_zero_flag.shape[0] != np.prod(x.shape):
        is_not_zero_flag = is_not_zero_flag[0: np.prod(x.shape)]
    is_not_zero_flag = is_not_zero_flag.reshape(x.shape)

    diff_accumor_x_bitwise = [None] * (len(x_bitwise) - 1) + [accum_x_bitwise[-1]]
    for i in range(len(x_bitwise) - 2, -1, -1):
        diff_accumor_x_bitwise[i] = accum_x_bitwise[i] - accum_x_bitwise[i + 1]
    a = diff_accumor_x_bitwise
    # print("a[0]@line 197=", a[0])
    # # -----------------------v = a^{-1}---------------------------------------
    # z = invert_one_hot(a, fixedpoint=careful_fixedpoint)
    # # print("z[0]@line 200=", z[0])
    # v = one_hot_bitwisePair2A(z, shape=x.shape, fixedpoint=careful_fixedpoint)
    # # print("v@line 202=", v)

    # --------------- u = (2a)^{-1/2} ---------------------------
    z = invert_one_hot(a, fixedpoint=careful_fixedpoint, rshift=1)
    u = one_hot_bitwisePair_sqrt2A(z, fixedpoint=careful_fixedpoint, shape=x.shape)
    # u = u.dup_with_precision(x.fixedpoint)
    # --------------------------------------------------
    # w = inverse_sqrt(_x * v) * u
    w = inverse_sqrt(_x, y=u)
    w = w.dup_with_precision(x.fixedpoint)
    w = select_share(is_not_zero_flag, w, 1 / eps * w.ones_like())
    return w


if __name__ == '__main__':
    start_local_server(config_file="../../../conf/config.json")
    a = [[1.0, 4.0, 1.0, 8.0], [0.0, 16.0, 64.0, 10000.0]]
    # a = [0.0]
    x = SharedPair(ownerL="L", ownerR="R", shape=[2, 4], fixedpoint=16)
    x.load_from_tf_tensor(tf.constant(a, dtype='float64'))
    y = A2bitwisePair(x)
    # z = one_hot_bitwisePair_sqrt2A(y, fixedpoint=16, shape=x.shape)
    r = careful_inverse_sqrt2(x, 1E-6)
    sess = tf.compat.v1.Session(target=StfConfig.target)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run([r.to_tf_tensor("R")]))
