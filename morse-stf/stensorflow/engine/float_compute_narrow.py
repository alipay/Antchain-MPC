#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : float_compare
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-24 19:54
   Description : description what the main function of this file
   export PYTHONPATH=$PYTHONPATH:/Users/qizhi.zqz/projects/morse-stf/morse-stf
"""
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, get_device
# from stensorflow.basic.protocol.compare import geq_SharedTensorBase_SharedTensorBase, \
#     greater_SharedTensorBase_SharedTensorBase, less_SharedTensorBase_SharedTensorBase, \
#     leq_SharedTensorBase_SharedTensorBase
from stensorflow.basic.protocol.msb import to_bool, bool_expansion, PrivateTensorBitwise, SharedTensorBitwise
from typing import Union
import time as time

def geq(x: tf.Tensor, y: tf.Tensor, x_owner, y_owner) -> SharedPair:
    """
     x>=y in Z/2^nZ iff  MSB(x+(2^n+2^n-y) in Z/2^{n+1}Z)==0  iff MSB(x+(2^n-1-y)+1 in Z/2^{n+1}Z)==1
    :param x:
    :param y:
    :param x_owner:
    :param y_owner:
    :return:
    """


    if x.dtype==tf.int16 or x.dtype==tf.int64 or x.dtype==tf.int32 or x.dtype==tf.float64 or x.dtype==tf.float32 or x.dtype==tf.float16:
        with tf.device(x_owner):
            list_x = bool_expansion(x)
            list_x = list(map(lambda xi: PrivateTensorBitwise(owner=x_owner, inner_value=SharedTensorBitwise(inner_value=xi)), list_x))

        with tf.device(y_owner):
            #list_y = bool_expansion(-tf.ones_like(y)-y)
            list_y = bool_expansion(y)
            list_y = list(map(lambda xi: PrivateTensorBitwise(owner=y_owner, inner_value=SharedTensorBitwise(inner_value=-tf.ones_like(xi)-xi)), list_y))

        carry = list_x[0] + list_y[0] + list_x[0] * list_y[0]
        
        for i in range(1,len(list_x)):
            carry= (list_x[i]+carry)*(list_y[i]+carry)-carry
        with tf.device(x_owner):
            xL = to_bool(carry.xL.inner_value)
            xL = tf.cast(xL, 'int64')
            xL = SharedTensorBase(inner_value=xL, module=2)
        with tf.device(y_owner):
            xR = to_bool(carry.xR.inner_value)
            xR = tf.cast(xR, 'int64')
            xR = SharedTensorBase(inner_value=xR, module=2)
        z = SharedPair(ownerL=x_owner, ownerR=y_owner, xL=xL, xR=xR, fixedpoint=0)
        z = z.reshape([-1])
        
        
        if z.shape[0] != np.prod(x.shape):
            z = z[0: np.prod(x.shape)]
        return z.reshape(x.shape)
    else:
        raise NotImplementedError


def leq(x: tf.Tensor, y: tf.Tensor, x_owner, y_owner) -> SharedPair:
    return geq(y, x, y_owner, x_owner)

def less(x: tf.Tensor, y: tf.Tensor, x_owner, y_owner) -> SharedPair:
    z = geq(x,y,x_owner,y_owner)
    return z.ones_like()-z

def greater(x: tf.Tensor, y: tf.Tensor, x_owner, y_owner) -> SharedPair:
    return less(y,x,y_owner,x_owner)


def compare_intfloat_intfloat(x_owner, x: tf.Tensor, y_owner, y: tf.Tensor, relation: str) -> SharedPairBase:
    x_owner=get_device(x_owner)
    y_owner=get_device(y_owner)
    if relation == "less":
        z = less(x, y, x_owner=x_owner, y_owner=y_owner)
    elif relation == "leq":
        z = leq(x, y, x_owner=x_owner, y_owner=y_owner)
    elif relation == "greater":
        z = greater(x, y, x_owner=x_owner, y_owner=y_owner)
    elif relation == "geq":
        z = geq(x, y, x_owner=x_owner, y_owner=y_owner)
    else:
        raise Exception("relation must be less, leq, greater or geq")
    return z


compare_int_int=compare_intfloat_intfloat

compare_float_float=compare_intfloat_intfloat
