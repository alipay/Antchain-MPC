#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : ot
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-02 15:45
   Description : description what the main function of this file
"""
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase
from stensorflow.basic.protocol.shift import cyclic_lshift


def assistant_ot(x: PrivateTensorBase, index: PrivateTensorBase, prf_flag=None,
                 compress_flag=None) -> SharedPairBase:
    if x.owner == index.owner:
        raise Exception("This function is for diffenrent owner")
    if x.shape[-1] != index.module:
        raise Exception("must have x.shape[-1]==index.module")
    if x.shape[:-1] != index.shape:
        raise Exception("must have x.shape[:-1]==index.shape")
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    z = cyclic_lshift(x=x, index=index, prf_flag=prf_flag, compress_flag=compress_flag)
    z = z[..., 0]
    return z