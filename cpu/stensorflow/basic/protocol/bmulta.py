#!/usr/bin/env python
# coding=utf-8
"""
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : pm1_act
   Author : Qizhi Zhang
   Email: zqz.math@gmail.ccom
   Create Time : 2022-12-24 11:01
   Description : (t, x) -> tx
"""

from stensorflow.basic.basic_class.pair import SharedPairBase
from stensorflow.basic.basic_class.share import SharedTensor, SharedTensorBase
from stensorflow.basic.basic_class.base import PrivateTensorBase
from stensorflow.random.random import get_seed, gen_rint64, gen_rint64_from_seed
from typing import Union
from stensorflow.exception.exception import StfCondException, StfEqualException, StfValueException, StfNoneException
from stensorflow.global_var import StfConfig
import tensorflow as tf
from stensorflow.basic.protocol.pm1_act import _pm1_act


def _bmulta(t: SharedTensor, x: SharedTensor) -> SharedTensor:
    # (t, x) -> (-1)^tx
    if t.module != 2:
        raise StfValueException("t.module", 2, t.module)
    inner_value = t.inner_value  * x.inner_value
    return SharedTensor(inner_value=inner_value, module=x.module)






def bmulta(t: SharedTensor, x: SharedTensor, t_owner, x_owner, RS_owner,
            prf_flag=None, compress_flag=None) \
        -> SharedPairBase:
    # Step 1.  generate t_adjoint, x_adjoint, b_t, b_x, s.t  t_adjoint x_adjoint = b_t + b_x
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    with tf.device(RS_owner):
        if prf_flag:
            seed_t = get_seed()
            seed_x = get_seed()
            seed_b = get_seed()
        else:
            seed_t = None
            seed_x = None
            seed_b = None
        t_adjoint = t.random_uniform_adjoint(seed=seed_t)
        x_adjoint = x.random_uniform_adjoint(seed=seed_x)
        b = _bmulta(t_adjoint, x_adjoint)
        b_t = b.random_uniform_adjoint(seed=seed_b)
        b_x = b - b_t
        if compress_flag and not prf_flag:
            t_adjoint_compress = t_adjoint.to_compress_tensor()

    # Step2. compute delta_t:=t-t_adjoint
    with tf.device(t_owner):
        if prf_flag:
            t_adjoint = t.random_uniform_adjoint(seed=seed_t)
            b_t = b.random_uniform_adjoint(seed=seed_b)
        elif compress_flag:
            t_adjoint.decompress_from(t_adjoint_compress)

        delta_t = t - t_adjoint
        if compress_flag:
            delta_t_compress = delta_t.to_compress_tensor()

    # Step 3. compute delta_x:=x-x_adjoint
    with tf.device(x_owner):
        if prf_flag:
            x_adjoint = x.random_uniform_adjoint(seed=seed_x)
        delta_x = x - x_adjoint

    # Step4. compute
    with tf.device(t_owner):
        y_t = _bmulta(t, delta_x) + _pm1_act(delta_t, b_t)

    # Step 5. compute
    with tf.device(x_owner):
        if compress_flag:
            delta_t.decompress_from(delta_t_compress)
        y_x = _bmulta(delta_t, x_adjoint) + _pm1_act(delta_t, b_x)
    return SharedPairBase(ownerL=t_owner, ownerR=x_owner, xL=y_t, xR=y_x, fixedpoint=0)



def bpair_multa(t: SharedPairBase, x: SharedTensor, x_owner, RS_owner,
                 prf_flag=None, compress_flag=None) \
        -> SharedPairBase:
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    if t.ownerL == x_owner and t.ownerR != x_owner:
        t = t.mirror()
    if t.ownerR == x_owner:
        pass
    else:
        raise StfCondException("t.ownerL==x_owner or t.ownerR==x.owner",
                               "t.ownerL={}, t.ownerR={}, x_owner={}".format(t.ownerL, t.ownerR, x_owner))
    with tf.device(x_owner):
        tR_mult_x = _bmulta(t.xR, x)
    return bpair_multa(t.xL, tR_mult_x, t_owner=t.ownerL, x_owner=x_owner, RS_owner=RS_owner,
                   prf_flag=prf_flag, compress_flag=compress_flag)






def bmulta_pair_pair(t: SharedPairBase, x: SharedPairBase, RS_owner, prf_flag=None,
                      compress_flag=None) -> SharedPairBase:
    """

    :param compress_flag:
    :param t:
    :param x:
    :param RS_owner:
    :return: (-1)^tx
    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if t.ownerL != x.ownerL or t.ownerR != x.ownerR:
        if t.ownerL != x.ownerR or t.ownerR != x.ownerL:
            raise Exception(
                "must have t.ownerL==x.ownerL and t.ownerR==x.ownerR or t.ownerL==x.ownerR and t.ownerR==x.ownerL")
        else:
            x = x.mirror()
    with tf.device(RS_owner):
        if prf_flag:
            seed_tL = get_seed()
            seed_tR = get_seed()
            seed_xL = get_seed()
            seed_xR = get_seed()
            seed_y = get_seed()
            tL_adjoint = t.to_SharedTensor_like().random_uniform_adjoint(seed_tL)
            tR_adjoint = t.to_SharedTensor_like().random_uniform_adjoint(seed_tR)
            xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xL)
            xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xR)
        else:
            tL_adjoint = t.to_SharedTensor_like().random_uniform_adjoint()
            tR_adjoint = t.to_SharedTensor_like().random_uniform_adjoint()
            xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
            xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
        y = _bmulta(tL_adjoint + tR_adjoint, xL_adjoint + xR_adjoint)
        if prf_flag:
            yL = y.random_uniform_adjoint(seed=seed_y)
        else:
            yL = y.random_uniform_adjoint()
        yR = y - yL
        if compress_flag:
            if not prf_flag:
                tL_adjoint_compressed = tL_adjoint.to_compress_tensor()
                tR_adjoint_compressed = tR_adjoint.to_compress_tensor()
    with tf.device(t.ownerL):
        if prf_flag:
            tL_adjoint = t.to_SharedTensor_like().random_uniform_adjoint(seed_tL)
            xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xL)
        elif compress_flag:
            tL_adjoint.decompress_from(tL_adjoint_compressed)
        delta_tL = t.xL - tL_adjoint
        delta_xL = x.xL - xL_adjoint
        zL = _pm1_act(t.xL, delta_xL)
        if compress_flag:
            delta_tL_compressed = delta_tL.to_compress_tensor()
    with tf.device(t.ownerR):
        if prf_flag:
            tR_adjoint = t.to_SharedTensor_like().random_uniform_adjoint(seed_tR)
            xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xR)
        elif compress_flag:
            tR_adjoint.decompress_from(tR_adjoint_compressed)
        delta_tR = t.xR - tR_adjoint
        delta_xR = x.xR - xR_adjoint
        zR = _pm1_act(t.xR, delta_xR)
        if compress_flag:
            delta_tR_compressed = delta_tR.to_compress_tensor()
    with tf.device(t.ownerL):
        if compress_flag:
            delta_tR_new = delta_tR.decompress_from_to_new(delta_tR_compressed)
            uL = _bmulta(t.xL, delta_xL + zR) + _bmulta(delta_tL+delta_tR_new, xL_adjoint) + _pm1_act(delta_tL+delta_tR_new, yL)
        else:
            uL = _bmulta(t.xL, delta_xL + zR) + _bmulta(delta_tL+delta_tR, xL_adjoint) + _pm1_act(delta_tL+delta_tR, yL)

    with tf.device(t.ownerR):
        if compress_flag:
            delta_tL_new = delta_tL.decompress_from_to_new(delta_tL_compressed)
            uR = _bmulta(t.xR, delta_xR + zL) + _bmulta(delta_tL_new+delta_tR, xR_adjoint) + _pm1_act(delta_tL_new+delta_tR, yR)
        else:
            uR = _bmulta(t.xR, delta_xR + zL) + _bmulta(delta_tL+delta_tR, xR_adjoint) + _pm1_act(delta_tL+delta_tR, yR)
    return SharedPairBase(ownerL=t.ownerL, ownerR=t.ownerR, xL=uL, xR=uR, fixedpoint=x.fixedpoint)

