#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : shift
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-12-01 20:09
   Description : description what the main function of this file
"""
from stensorflow.basic.basic_class.base import SharedPairBase
from stensorflow.basic.basic_class.base import PrivateTensorBase
from stensorflow.random.random import get_seed
import tensorflow as tf
from stensorflow.global_var import StfConfig


def cyclic_lshift(index: PrivateTensorBase, x: PrivateTensorBase, prf_flag=None, compress_flag=None) -> SharedPairBase:
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
    with tf.device(StfConfig.RS[0]):
        # STEP 1 :  Random Server generate index_adjoint and x_adjoint, y_L, y_R, s.t
        #  L_{index_adjoint} x_adjoint = y_L + y_R
        if prf_flag:
            seed_index = get_seed()
            seed_x = get_seed()
            seed_y = get_seed()
        else:
            seed_index = None
            seed_x = None
            seed_y = None

        index_adjoint = index.to_SharedTensor_like().random_uniform_adjoint(seed_index)
        x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_x)
        y = x_adjoint.lshift(index_adjoint.inner_value)
        yL = y.random_uniform_adjoint(seed_y)
        yR = y - yL

        if compress_flag:
            if not prf_flag:
                x_adjoint_compress = x_adjoint.to_compress_tensor()
                index_adjoint_compress = index_adjoint.to_compress_tensor()
                yL_compress = yL.to_compress_tensor()
            yR_compress = yR.to_compress_tensor()

    # STEP 2: x.owner compute delta_x=x-x_adjoint
    with tf.device(x.owner):
        if prf_flag:
            x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_x)
        elif compress_flag:
            x_adjoint.decompress_from(x_adjoint_compress, shape=x_adjoint.shape)
        delta_x = x.to_SharedTensor() - x_adjoint
        if compress_flag:
            delta_x_compress = delta_x.to_compress_tensor()

    # Step 3. index.owner compute delta_i = index - index_adjoint
    #         and zL:=L_{index}(delta_x)+L_{delta_i}(y_L)
    with tf.device(index.owner):
        if prf_flag:
            index_adjoint = index.to_SharedTensor_like().random_uniform_adjoint(seed_index)
            yL = y.random_uniform_adjoint(seed_y)
        elif compress_flag:
            index_adjoint.decompress_from(index_adjoint_compress, shape=index_adjoint.shape)
            yL.decompress_from(yL_compress, yL.shape)
        delta_i = index.to_SharedTensor() - index_adjoint

        if compress_flag:
            delta_x.decompress_from(delta_x_compress, shape=delta_x.shape)
            delta_i_compress = delta_i.to_compress_tensor()

        zL = delta_x.lshift(index.inner_value) + yL.lshift(delta_i.inner_value)

    # Step 4.  x.owner compute zR:=L_{delta_i}(yR)
    with tf.device(x.owner):
        if compress_flag:
            delta_i.decompress_from(delta_i_compress, delta_i.shape)
            yR.decompress_from(yR_compress, yR.shape)
        zR = yR.lshift(delta_i.inner_value)
    z = SharedPairBase(ownerL=index.owner, ownerR=x.owner, xL=zL, xR=zR, fixedpoint=x.fixedpoint)
    return z
