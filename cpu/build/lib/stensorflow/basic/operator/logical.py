#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : logical
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 11:42
   Description : description what the main function of this file
"""
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase
from stensorflow.basic.operator.arithmetic import mul, add
from typing import Union
from stensorflow.global_var import StfConfig


def check_type_module(x, y):
    if isinstance(x, PrivateTensorBase) and isinstance(y, PrivateTensorBase):
        if x.module != 2 or y.module != 2:
            raise Exception("both x and y must have module 2")
    elif isinstance(x, PrivateTensorBase) and isinstance(y, SharedPairBase):
        if x.module != 2 or y.xL.module != 2:
            raise Exception("both x and y must have module 2")
    elif isinstance(x, SharedPairBase) and isinstance(y, PrivateTensorBase):
        if x.xL.module != 2 or y.module != 2:
            raise Exception("both x and y must have module 2")
    elif isinstance(x, SharedPairBase) and isinstance(y, SharedPairBase):
        if x.xL.module != 2 or y.xL.module != 2:
            raise Exception("both x and y must have module 2")
    else:
        raise Exception("x, y must be PrivateTensorBase or SharedPairBase")


def and_op(x, y, prf_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    check_type_module(x, y)
    return mul(x, y, prf_flag=prf_flag)


def xor_op(x, y):
    check_type_module(x, y)
    return add(x, y)


def or_op(x, y, prf_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    check_type_module(x, y)
    return add(add(x, y), mul(x, y, prf_flag=prf_flag))


def not_op(x: Union[PrivateTensorBase, SharedPairBase]):
    if isinstance(x, PrivateTensorBase):
        if x.module != 2:
            raise Exception("must have x.module == 2")
    elif isinstance(x, SharedPairBase):
        if x.xL.module != 2:
            raise Exception("must have x.module == 2")
    else:
        raise Exception("x must be a PrivateTensorBase or SharedPairBase")
    return x.ones_like() - x


def nxor_op(x, y):
    z = xor_op(x, y)
    return not_op(z)
