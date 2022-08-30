#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : argmax
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/8/4 下午3:49
   Description : description what the main function of this file
"""


from stensorflow.basic.operator.selectshare import select_share
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.share import SharedTensorBase
from stensorflow.basic.basic_class.private import PrivateTensor
import tensorflow as tf

def argmax(x: SharedPair, axis, module=None, return_max=False):
    if module is None:
        module = x.shape[axis]
    dim_axis = x.shape[axis]
    if dim_axis == 1:
        zeros = x.zeros_like()
        zeros.xL.module = module
        zeros.xR.module = module
        zeros = zeros.zeros_like()
        if return_max:
            return (zeros, x)
        else:
            return zeros
    else:
        print("dim_axis=", dim_axis)
        print("axis=", axis)
        x0, x1 = x.split(size_splits=[dim_axis // 2, dim_axis - dim_axis // 2], axis=axis)

        argmax0, max0 = argmax(x0, axis, module=module, return_max=True)
        argmax1, max1 = argmax(x1, axis, module=module, return_max=True)
        argmax1 = argmax1 + (dim_axis // 2) * argmax1.ones_like()
        s = (max0 > max1)
        print("s=",s)
        print("argmax0=", argmax0)
        print("zrgmax1=", argmax1)
        argmax_ = select_share(s, argmax0, argmax1)
        argmax_ = SharedPair.from_SharedPairBase(argmax_)
        if return_max:
            max_ = select_share(s, max0, max1)
            max_ = SharedPair.from_SharedPairBase(max_)
            return (argmax_, max_)
        else:
            return argmax_




def argmin(x: SharedPair, axis, module=None, return_min=False):
   return argmax(SharedPair.from_SharedPairBase(-x), axis, module=module, return_max=return_min)