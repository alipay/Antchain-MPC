#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : sigmoid
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 11:42
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.share import sin2pi as sin2pi_share, cos2pi as cos2pi_share
import tensorflow as tf
from stensorflow.global_var import StfConfig
from typing import Union
import numpy as np


def sigmoid_poly(x: SharedPair):
    """A Chebyshev polynomial approximation of the sigmoid function."""

    w0 = 0.5
    w1 = 0.2159198015
    w3 = -0.0082176259
    w5 = 0.0001825597
    w7 = -0.0000018848
    w9 = 0.0000000072

    x1 = x
    x2 = (x1 * x).dup_with_precision(x.fixedpoint)
    x3 = (x2 * x).dup_with_precision(x.fixedpoint)
    x5 = (x2 * x3).dup_with_precision(x.fixedpoint)
    x7 = (x2 * x5).dup_with_precision(x.fixedpoint)
    x9 = (x2 * x7).dup_with_precision(x.fixedpoint)

    y1 = w1 * x1
    y3 = w3 * x3
    y5 = w5 * x5
    y7 = w7 * x7
    y9 = w9 * x9

    z = y9 + y7 + y5 + y3 + y1 + tf.constant(w0)

    return z


def sigmoid_poly_minmax(x: SharedPair):
    """A minmax polynomial approximation of the sigmoid function.
    """

    w0 = 0.5
    w1 = 0.197
    w3 = -0.004
    x1 = x
    x2 = x1 * x
    x3 = x2 * x
    y1 = w1 * x1
    y3 = w3 * x3

    z = y3 + y1 + tf.constant(w0)
    # z = y7 + y5 + y3 + y1 + w0

    return z


def sin2pi(x: SharedPair, T: int = 1, k: Union[int, tf.Tensor] = None) -> SharedPair:
    # sin(2kpix/T)
    # print("x.xL.shape=", x.xL.shape)
    # print("x.xR.shape=", x.xR.shape)
    n = int(np.log2(T))

    if 1 << n != T:
        raise Exception("T must be a power of 2")

    if k is None:
        k = 1

    if isinstance(k, int):
        pass
    elif isinstance(k, tf.Tensor):
        if k.dtype in [tf.dtypes.int8, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64]:
            pass
    else:
        raise Exception("the type of k is error")

    with tf.device(x.ownerL):
        yL = tf.stack([sin2pi_share(x.xL, x.fixedpoint + n, k), cos2pi_share(x.xL, x.fixedpoint + n, k)], axis=-1)

    with tf.device(x.ownerR):
        yR = tf.stack([cos2pi_share(x.xR, x.fixedpoint + n, k), sin2pi_share(x.xR, x.fixedpoint + n, k)], axis=-1)

    if StfConfig.parties == 3:
        with tf.device(x.ownerL):
            yL = tf.expand_dims(yL, axis=-2)

            zL = PrivateTensor(owner=x.ownerL)
            zL.load_from_tf_tensor(yL)
        with tf.device(x.ownerR):
            yR = tf.expand_dims(yR, axis=-1)

            zR = PrivateTensor(owner=x.ownerR)
            zR.load_from_tf_tensor(yR)

        result = (zL @ zR).dup_with_precision(new_fixedpoint=StfConfig.default_fixed_point)
        result = result.squeeze(axis=[-1, -2])
    else:
        with tf.device(x.ownerL):
            zL = PrivateTensor(owner=x.ownerL)
            zL.load_from_tf_tensor(yL)
        with tf.device(x.ownerR):
            zR = PrivateTensor(owner=x.ownerR)
            zR.load_from_tf_tensor(yR)

        result = (zL * zR).reduce_sum(axis=[-1]).dup_with_precision(new_fixedpoint=StfConfig.default_fixed_point)
    return result












def sigmoid_sin(x: SharedPair, M=16):
    """Fourier series approximation of the sigmoid function.
    https://arxiv.org/pdf/2109.11726.pdf """
    if 1 << int(np.log2(M)) != M:
        raise Exception("M must be a power of 2")
    term = 6
    sample_num = 256
    X = np.linspace(-M, M, sample_num, endpoint=False)  # -M to+M的256个值

    sigmoid = 1 / (1 + np.exp(-X))
    sm5 = sigmoid - 0.5

    sm5_odd = sm5 * 1.0
    sm5_odd[0] = 0
    F = np.fft.fft(sm5_odd)
    a = F[0:term].imag
    # a = tf.constant(a, dtype='float32', shape=[term]+[1]*len(x.shape))
    a = np.reshape(a, newshape=[term] + [1] * len(x.shape))
    a = tf.constant(a, dtype='float32')
    integers = np.reshape(range(term), newshape=[term] + [1] * len(x.shape))
    integers = tf.constant(integers, dtype='int64')
    x = x.expend_dims(axis=[0])
    s = sin2pi(x - M, T=2 * M, k=integers)
    y = -a / 128 * s
    y = y.reduce_sum(axis=[0])
    y = y + 0.5
    return y




def sigmoid_idea(x: SharedPair, M=16):
    """The idea sigmoid"""
    y=tf.sigmoid(x.to_tf_tensor("R"))
    z = SharedPair(ownerL="L", ownerR="R", shape=y.shape)
    z.load_from_tf_tensor(y)
    return z

def sigmoid_local(x: PrivateTensor):
    z = PrivateTensor(owner=x.owner)
    with tf.device(x.owner):
        y = tf.sigmoid(x.to_tf_tensor())
        z.load_from_tf_tensor(y)
    return z
