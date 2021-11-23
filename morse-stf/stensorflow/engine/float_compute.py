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
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, get_device
from stensorflow.basic.protocol.compare import geq_SharedTensorBase_SharedTensorBase, \
    greater_SharedTensorBase_SharedTensorBase, less_SharedTensorBase_SharedTensorBase, \
    leq_SharedTensorBase_SharedTensorBase
from typing import Union
import time as time

def to_bits(x: tf.Tensor, little_endian=True) -> tf.Tensor:
    # if x.dtype != tf.int64:
    #     raise Exception("must have x.dtype==tf.int64")
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


def f2bits_keep_order(x: tf.Tensor) -> tf.Tensor:
    y = to_bits(x, little_endian=False)
    y_shape = y.shape.as_list()
    y = tf.reshape(y, [-1, y_shape[-1]])
    sign = y[:, 0:1]
    y = (y + sign) % 2
    z = tf.concat([1 - sign, y[:, 1:]], axis=1)
    return tf.reshape(z, y_shape)


def compare_float_float(x_owner, x: tf.Tensor, y_owner, y: tf.Tensor, relation: str) -> SharedPairBase:
    x_owner=get_device(x_owner)
    y_owner=get_device(y_owner)
    
    with tf.device(x_owner):
        x2b = f2bits_keep_order(x)
        
        x2b = SharedTensorBase(inner_value=x2b, module=2)
    with tf.device(y_owner):
        y2b = f2bits_keep_order(y)
        
        y2b = SharedTensorBase(inner_value=y2b, module=2)
    if relation == "less":
        z = less_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "leq":
        z = leq_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "greater":
        z = greater_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "geq":
        z = geq_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    else:
        raise Exception("relation must be less, leq, greater or geq")
    return z


def compare_int_int(x_owner, x: tf.Tensor, y_owner, y: tf.Tensor, relation: str) -> SharedPairBase:
    x_owner=get_device(x_owner)
    y_owner=get_device(y_owner)
    with tf.device(x_owner):
        x2b = to_bits(x, little_endian=False)
        
        x2b = SharedTensorBase(inner_value=x2b, module=2)
    with tf.device(y_owner):
        y2b = to_bits(y, little_endian=False)
        
        y2b = SharedTensorBase(inner_value=y2b, module=2)



    if relation == "less":
        z = less_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "leq":
        z = leq_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "greater":
        z = greater_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    elif relation == "geq":
        z = geq_SharedTensorBase_SharedTensorBase(x2b, y2b, x_owner=x_owner, y_owner=y_owner)
    else:
        raise Exception("relation must be less, leq, greater or geq")
    return z