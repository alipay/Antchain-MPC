#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : module_transform
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-08 19:35
   Description : modular transform: https://eprint.iacr.org/2021/857
"""

from stensorflow.basic.basic_class.base import SharedPairBase, SharedTensorBase
from stensorflow.random.random import get_seed
from stensorflow.global_var import StfConfig
import tensorflow as tf


def module_transform_withoutPRF(a: SharedPairBase, new_module: int, compress_flag=None):
    if a.fixedpoint > 0:
        raise Exception("must have a.fixedpoint==0")
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    old_module = a.xL.module
    if old_module == 2:
        with tf.device(StfConfig.RS[0]):
            uL = a.to_SharedTensor_like().random_uniform_adjoint()
            uR = a.to_SharedTensor_like().random_uniform_adjoint()
            u = uL + uR
            u.module = new_module
            bL = u.random_uniform_adjoint()
            bR = u - bL

            if compress_flag:
                uL_compress = uL.to_compress_tensor()
                uR_compress = uR.to_compress_tensor()
                bL_compress = bL.to_compress_tensor()
                bR_compress = bR.to_compress_tensor()

        with tf.device(a.ownerL):
            if compress_flag:
                uL.decompress_from(uL_compress, uL.shape)
                bL.decompress_from(bL_compress, bL.shape)
            zL = a.xL - uL
            if compress_flag:
                zL_compress = zL.to_compress_tensor()
        with tf.device(a.ownerR):
            if compress_flag:
                uR.decompress_from(uR_compress, uR.shape)
                bR.decompress_from(bR_compress, bR.shape)
            zR = a.xR - uR
            if compress_flag:
                zR_compress = zR.to_compress_tensor()
                zL.decompress_from(zL_compress, zL.shape)
            z = zL + zR
            yR = tf.where(tf.cast(z.inner_value, 'bool'), x=-bR.inner_value, y=bR.inner_value)
            yR = SharedTensorBase(inner_value=yR + z.inner_value, module=new_module)
        with tf.device(a.ownerL):
            if compress_flag:
                zR.decompress_from(zR_compress, zR.shape)
            z = zL + zR
            yL = tf.where(tf.cast(z.inner_value, 'bool'), x=-bL.inner_value, y=bL.inner_value)
            yL = SharedTensorBase(inner_value=yL, module=new_module)

        return SharedPairBase(ownerL=a.ownerL, ownerR=a.ownerR, xL=yL, xR=yR, fixedpoint=0)
    else:
        raise NotImplementedError


def module_transform_withPRF(a: SharedPairBase, new_module: int, compress_flag=None):
    if a.fixedpoint > 0:
        raise Exception("must have a.fixedpoint==0")
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    old_module = a.xL.module
    if old_module == 2:
        with tf.device(StfConfig.RS[0]):
            seed_aL = get_seed()
            seed_aR = get_seed()
            seed_b = get_seed()
            aL_adjoint = a.to_SharedTensor_like().random_uniform_adjoint(seed_aL)
            aR_adjoint = a.to_SharedTensor_like().random_uniform_adjoint(seed_aR)
            b = aL_adjoint + aR_adjoint
            b.module = new_module
            bL = b.random_uniform_adjoint(seed_b)
            bR = b - bL
            if compress_flag:
                bR_compress = bR.to_compress_tensor()
        with tf.device(a.ownerL):
            aL_adjoint_onL = a.to_SharedTensor_like().random_uniform_adjoint(seed_aL)
            delta_aL = a.xL - aL_adjoint_onL
            if compress_flag:
                delta_aL_compress = delta_aL.to_compress_tensor()
        with tf.device(a.ownerR):
            aR_adjoint_onR = a.to_SharedTensor_like().random_uniform_adjoint(seed_aR)
            delta_aR = a.xR - aR_adjoint_onR
            if compress_flag:
                bR.decompress_from(bR_compress, bR.shape)
                delta_aR_compress = delta_aR.to_compress_tensor()
                delta_aL_onR = delta_aL.decompress_from_to_new(delta_aL_compress)
            else:
                delta_aL_onR = delta_aL.identity()
            delta_a_onR = delta_aL_onR + delta_aR
            yR = tf.where(tf.cast(delta_a_onR.inner_value, 'bool'), x=-bR.inner_value, y=bR.inner_value)
            yR = SharedTensorBase(inner_value=yR + delta_a_onR.inner_value, module=new_module)
        with tf.device(a.ownerL):
            if compress_flag:
                delta_aR_onL = delta_aR.decompress_from_to_new(delta_aR_compress)
            else:
                delta_aR_onL = delta_aR.identity()
            delta_a_onL = delta_aL + delta_aR_onL
            yL = tf.where(tf.cast(delta_a_onL.inner_value, 'bool'), x=-bL.inner_value, y=bL.inner_value)
            yL = SharedTensorBase(inner_value=yL, module=new_module)
        return SharedPairBase(ownerL=a.ownerL, ownerR=a.ownerR, xL=yL, xR=yR, fixedpoint=0)
    else:
        raise NotImplementedError


def module_transform(a: SharedPairBase, new_module: int, prf_flag=None, compress_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    if prf_flag:
        return module_transform_withPRF(a, new_module, compress_flag)
    else:
        return module_transform_withoutPRF(a, new_module, compress_flag)
