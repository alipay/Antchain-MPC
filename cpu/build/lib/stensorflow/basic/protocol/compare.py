#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : compare
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2020-05-14 11:44
   Description : compare : https://eprint.iacr.org/2021/857
"""

from stensorflow.basic.protocol.fnz import fnz_v3 as fnz
from stensorflow.basic.protocol.module_transform import module_transform
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from stensorflow.global_var import StfConfig
from stensorflow.basic.protocol.ot import assistant_ot
from sympy.ntheory.modular import isprime
import tensorflow as tf


def get_next_prime(n: int):
    while 1:
        if isprime(n):
            return n
        else:
            n += 1


def less_SharedTensorBase_SharedTensorBase(x: SharedTensorBase, y: SharedTensorBase, x_owner,
                                           y_owner) -> SharedPairBase:
    """

    :param x, y:  SharedPair with module=2, shape=[n0, n1, ..., nk]
    :return:  SharedPair with module=2,  shape=[n0, n1, ...nk-1]   x<y in lexicographical order
    """

    if x.module != 2 or y.module != 2:
        raise Exception("must have x.module==2 and y.module==2")
    if x.shape != y.shape:
        raise Exception("must have x.shape==y.shape")
    if x_owner == y_owner:
        raise Exception("must have x_owner!=y_owner")

    p = get_next_prime(x.shape[-1] + 3)

    z = SharedPairBase(ownerL=x_owner, ownerR=y_owner, xL=x, xR=y, fixedpoint=0)
    zp = module_transform(z, p)

    paddings = [[0, 0]] * len(x.shape)
    paddings[-1] = [0, 1]  # add (0, 1) following z[n-1]
    with tf.device(x_owner):
        x_extp = zp.xL.pad(paddings=paddings, constant_values=0)
    with tf.device(y_owner):
        y_ext = y.pad(paddings=paddings)
        y_extp = zp.xR.pad(paddings=paddings, constant_values=1)
    z_extp = SharedPairBase(ownerL=x_owner, ownerR=y_owner, xL=x_extp, xR=y_extp, fixedpoint=0)

    first_nonzero_bit = fnz(z_extp)

    with tf.device(x_owner):
        first_nonzero_bitL = PrivateTensorBase(owner=x_owner, fixedpoint=0,
                                               inner_value=first_nonzero_bit.xL.inner_value,
                                               module=first_nonzero_bit.xL.module)
    with tf.device(y_owner):
        shifted_y_ext = y_ext.lshift(first_nonzero_bit.xR.inner_value)
        p_shifted_y_ext = PrivateTensorBase(owner=y_owner, fixedpoint=0, inner_value=shifted_y_ext.inner_value,
                                            module=shifted_y_ext.module)
    xly = assistant_ot(p_shifted_y_ext, first_nonzero_bitL, prf_flag=StfConfig.prf_flag,
                       compress_flag=StfConfig.compress_flag)
    return xly


def leq_SharedTensorBase_SharedTensorBase(x: SharedTensorBase, y: SharedTensorBase, x_owner, y_owner) -> SharedPairBase:
    """

    :param x, y:  SharedPair with module=2, shape=[n0, n1, ..., nk]
    :return:  SharedPair with module=2,  shape=[n0, n2, ...nk-1]   x<=y in lexicographical order
    """

    if x.module != 2 or y.module != 2:
        raise Exception("must have x.module==2 and y.module==2")
    if x.shape != y.shape:
        raise Exception("must have x.shape==y.shape")
    if x_owner == y_owner:
        raise Exception("must have x_owner!=y_owner")

    p = get_next_prime(x.shape[-1] + 3)  # refer to

    z = SharedPairBase(ownerL=x_owner, ownerR=y_owner, xL=x, xR=y, fixedpoint=0)
    zp = module_transform(z, p)
    paddings = [[0, 0]] * len(x.shape)
    paddings[-1] = [0, 1]  # add (1, 0) following x[n-1]
    with tf.device(x_owner):
        x_extp = zp.xL.pad(paddings=paddings, constant_values=0)
    with tf.device(y_owner):
        y_ext = y.pad(paddings=paddings, constant_values=1)
        y_extp = zp.xR.pad(paddings=paddings, constant_values=1)
    z_extp = SharedPairBase(ownerL=x_owner, ownerR=y_owner, xL=x_extp, xR=y_extp, fixedpoint=0)

    first_nonzero_bit = fnz(z_extp)

    with tf.device(y_owner):

        shifted_y_ext = y_ext.lshift(first_nonzero_bit.xR.inner_value)
        _shifted_y_ext = PrivateTensorBase(owner=y_owner, fixedpoint=0, inner_value=shifted_y_ext.inner_value,
                                           module=shifted_y_ext.module)

    with tf.device(x_owner):

        first_nonzero_bitL = PrivateTensorBase(owner=x_owner, fixedpoint=0,
                                               inner_value=first_nonzero_bit.xL.inner_value,
                                               module=first_nonzero_bit.xL.module)

    xly = assistant_ot(_shifted_y_ext, first_nonzero_bitL, prf_flag=StfConfig.prf_flag,
                       compress_flag=StfConfig.compress_flag)
    return xly


def greater_SharedTensorBase_SharedTensorBase(x: SharedTensorBase, y: SharedTensorBase, x_owner,
                                              y_owner) -> SharedPairBase:
    """

    :param x, y:  SharedPair with module=2, shape=[n0, n1, ..., nk]
    :return:  SharedPair with module=2,  shape=[n0, n2, ...nk-1]   x>y in lexicographical order
    """

    return less_SharedTensorBase_SharedTensorBase(y, x, y_owner, x_owner)


def geq_SharedTensorBase_SharedTensorBase(x: SharedTensorBase, y: SharedTensorBase, x_owner, y_owner) -> SharedPairBase:
    """
    :param x, y:  SharedPair with module=2, shape=[n0, n1, ..., nk]
    :return:  SharedPair with module=2,  shape=[n0, n2, ...nk-1]   x>=y in lexicographical order
    """
    return leq_SharedTensorBase_SharedTensorBase(y, x, y_owner, x_owner)
