#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : base
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 19:58
   Description : description what the main function of this file
"""
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from stensorflow.exception.exception import StfTypeException
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.random.random import get_seed


def BM_PrivateTensor_PrivateTensor(x: PrivateTensorBase, y: PrivateTensorBase, f, prf_flag=None,
                                   x_adjoint: SharedTensorBase=None, y_adjoint: SharedTensorBase=None,
                                   u0: SharedTensorBase=None, u1: SharedTensorBase=None) -> SharedPairBase:
    # f: (SharedTensorBase, SharedTensorBase)->SharedTensorBase
    adjoint_is_None = (x_adjoint is None or y_adjoint is None or u0 is None or u1 is None)
    if adjoint_is_None:
        assert (x_adjoint is None and y_adjoint is None and u0 is None and u1 is None)
        if prf_flag is None:
            prf_flag = StfConfig.prf_flag
        if x.module != y.module:
            raise Exception("must have x.module==y.module")
        if x.owner == y.owner:
            raise Exception("must have x.owner != y.owner")
        with tf.device(StfConfig.RS[0]):
            if prf_flag:
                seedx = get_seed()
                seedy = get_seed()
                seedu = get_seed()
                x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed=seedx)
                y_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed=seedy)
            else:
                x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
                y_adjoint = y.to_SharedTensor_like().random_uniform_adjoint()
            u = f(x_adjoint, y_adjoint)
            if not isinstance(u, SharedTensorBase):
                raise StfTypeException("u", "SharedTensorBase", u.tupe)
            if prf_flag:
                u0 = u.random_uniform_adjoint(seed=seedu)
            else:
                u0 = u.random_uniform_adjoint()
            u1 = u - u0
    with tf.device(x.owner):
        if prf_flag and x_adjoint is None:
            x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed=seedx)
        delta_x = x.to_SharedTensor() - x_adjoint
    with tf.device(y.owner):
        if prf_flag and y_adjoint is None:
            y_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed=seedy)
        delta_y = y.to_SharedTensor() - y_adjoint
        v1 = f(delta_x, y.to_SharedTensor()) + u1
    with tf.device(x.owner):
        v0 = f(x_adjoint, delta_y) + u0
    return SharedPairBase(xL=v0, xR=v1, ownerL=x.owner, ownerR=y.owner, fixedpoint=x.fixedpoint + y.fixedpoint)


def BM_PrivateTensor_SharedPair(x: PrivateTensorBase, y: SharedPairBase, f, prf_flag=None,
                                x_adjoint=None, y_adjoint=None, u0=None, u1=None) -> SharedPairBase:
    # f: (PrivateTensorBase, SharedTensorBase)->SharedTensorBase

    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if x.module != y.xL.module:
        raise Exception("must have x.module==y.module")
    if y.ownerL == y.ownerR:
        raise Exception("must have y.ownerL!=y.ownerR")
    if x.owner == y.ownerL:
        adjoint_is_None = (x_adjoint is None or y_adjoint is None or u0 is None or u1 is None)
        assert (x_adjoint is None and y_adjoint is None and u0 is None and u1 is None)
        if adjoint_is_None:
            with tf.device(StfConfig.RS[0]):
                if prf_flag:
                    seedx = get_seed()
                    seedy = get_seed()
                    seedu = get_seed()
                    x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed=seedx)
                    y_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed=seedy)
                else:
                    x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
                    y_adjoint = y.xR.random_uniform_adjoint()
                u = f(x_adjoint, y_adjoint)
                if not isinstance(u, SharedTensorBase):
                    raise StfTypeException("u", "SharedTensorBase", u.tupe)
                if prf_flag:
                    u0 = u.random_uniform_adjoint(seed=seedu)
                else:
                    u0 = u.random_uniform_adjoint()
                u1 = u - u0
        with tf.device(x.owner):
            if prf_flag and x_adjoint is None:
                x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed=seedx)
            delta_x = x.to_SharedTensor() - x_adjoint
        with tf.device(y.ownerR):
            if prf_flag and y_adjoint is None:
                y_adjoint = y.xR.random_uniform_adjoint(seed=seedy)
            delta_y = y.xR - y_adjoint
            v1 = f(delta_x, y.xR)
            v1 = v1 + u1
        with tf.device(x.owner):
            v0 = f(x_adjoint, delta_y) + u0
            xy0 = f(x.to_SharedTensor(), y.xL)
        xy = SharedPairBase(xL=v0 + xy0, xR=v1, ownerL=y.ownerL, ownerR=y.ownerR,
                            fixedpoint=x.fixedpoint + y.fixedpoint)
    elif x.owner == y.ownerR:
        y = y.mirror()
        xy = BM_PrivateTensor_SharedPair(x, y, f, prf_flag=prf_flag, x_adjoint=x_adjoint, y_adjoint=y_adjoint, u0=u0, u1=u1)
        xy = xy.mirror()
    else:
        raise Exception("must have x.owner==y.ownerL or x.owner==y.ownerR")
    return xy


def BM_SharedPair_PrivateTensor(x: SharedPairBase, y: PrivateTensorBase, f, prf_flag=None,
                                x_adjoint=None, y_adjoint=None, u0=None, u1=None) -> SharedPairBase:
    return BM_PrivateTensor_SharedPair(y, x, lambda a, b: f(b, a), prf_flag, y_adjoint, x_adjoint, u1, u0)


def BM_SharedPair_SharedPair(x: SharedPairBase, y: SharedPairBase, f, prf_flag=None,
                             xL_adjoint=None, xR_adjoin=None, yL_adjoint=None, yR_adjoint=None,
                             uL=None, uR=None) -> SharedPairBase:
    # f: (SharedTensorBase, SharedTensorBase)->SharedTensorBase
    adjoint_is_None = (xL_adjoint is None or xR_adjoin is None or yL_adjoint is None
                       or yR_adjoint is None or uL is None or uR is None)
    if adjoint_is_None:
        assert (xL_adjoint is None and xR_adjoin is None and yL_adjoint is None
                       and yR_adjoint is None and uL is None and uR is None)
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag

    if x.xL.module != y.xL.module:
        raise Exception("must have x.xL.module==y.xL.module")
    if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
        y = y.mirror()
    if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
        raise Exception("must have x.ownerL==y.ownerL and x.ownerR==y.ownerR")
    if adjoint_is_None:
        with tf.device(StfConfig.RS[0]):
            if prf_flag:
                seed_xL = get_seed()
                seed_xR = get_seed()
                seed_yL = get_seed()
                seed_yR = get_seed()
                seed_u = get_seed()
                xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xL)
                xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xR)
                yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yL)
                yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yR)
            else:
                xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
                yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint()
                xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint()
                yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint()
            u = f(xL_adjoint + xR_adjoint, yL_adjoint + yR_adjoint)  # SharedTensor
            if not isinstance(u, SharedTensorBase):
                raise StfTypeException("u", "SharedTensorBase", u.tupe)
            if prf_flag:
                uL = u.random_uniform_adjoint(seed_u)
            else:
                uL = u.random_uniform_adjoint()
            uR = u - uL
    with tf.device(x.ownerL):
        if prf_flag and adjoint_is_None:
            xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xL)
            yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yL)
            uL = u.random_uniform_adjoint(seed_u)
        delta_xL = x.xL - xL_adjoint
        delta_yL = y.xL - yL_adjoint
    with tf.device(y.ownerR):
        if prf_flag and adjoint_is_None:
            xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xR)
            yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yR)
        delta_xR = x.xR - xR_adjoint
        delta_yR = y.xR - yR_adjoint
    with tf.device(x.ownerL):
        vL = f(delta_xL + delta_xR, delta_yL + delta_yR) + f(xL_adjoint, delta_yL + delta_yR) + \
            f(delta_xL + delta_xR, yL_adjoint) + uL
    with tf.device(x.ownerR):
        vR = f(xR_adjoint, delta_yL + delta_yR) + f(delta_xL + delta_xR, yR_adjoint) + uR
    return SharedPairBase(xL=vL, xR=vR, ownerL=x.ownerL, ownerR=x.ownerR, fixedpoint=x.fixedpoint + y.fixedpoint)
