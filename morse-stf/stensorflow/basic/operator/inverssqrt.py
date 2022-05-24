#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : inverssqrt
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/5/16 下午4:39
   Description : description what the main function of this file
"""


from stensorflow.basic.basic_class.pair import SharedPair
import tensorflow as tf
from stensorflow.exception.exception import StfTypeException
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from typing import Union
import numpy as np

def invers_sqrt(x: SharedPair, eps=0.0):
    # y = 1 / (tf.sqrt(x.to_tf_tensor("R")) + eps)
    y = 1 / (tf.sqrt(x.to_tf_tensor("R") + eps*eps))
    z = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, shape=y.shape)
    z.load_from_tf_tensor(y)
    return z

# def invers_sqrt(x: Union[SharedPair, PrivateTensor], eps=0.0):
#     # 1/sqrt(x+eps**2)
#     if isinstance(x, SharedPair):
#         y = x.ones_like()
#         for _ in range(StfConfig.inv_sqrt_iter_num):
#             if eps == 0.0:
#                 y = 1.5 * y - 0.5 * x * y ** 3
#             else:
#                 y = 1.5 * y - 0.5 * x * y ** 3 - 0.5 * (eps * y) ** 2 * y
#         return y
#     elif isinstance(x, PrivateTensor):
#         with tf.device(x.owner):
#             x = x.to_tf_tensor()
#             y = 1/(tf.sqrt(x+eps ** 2))
#             y = tf.cast(y * (2 ** x.fixedpoint), 'int64')
#             z = PrivateTensor(owner=x.owner, fixedpoint=x.fixedpoint,
#                               inner_value=y, module=x.module, op_map=x.op_map)
#             return z
#     else:
#         raise StfTypeException("x", "SharedPair or PrivateTensor", type(x))



def invers_sqrt_diff_eq(x: Union[SharedPair, PrivateTensor], x0=1E-8, y0=1E4, iter=10):
    y = y0
    h = (x-x0)/iter

    for _ in range(iter):
        dydx = -0.5 * (y ** 3)
        y = y + dydx * h
    return y