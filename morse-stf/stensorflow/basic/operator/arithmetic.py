#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : arithmetic.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 11:42
   Description : description what the main function of this file
"""
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
import numpy as np
from stensorflow.exception.exception import StfValueException, StfEqualException, StfCondException
import tensorflow as tf
from typing import Union
from stensorflow.global_var import StfConfig
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_SharedPair, BM_PrivateTensor_PrivateTensor, \
    BM_SharedPair_SharedPair, BM_SharedPair_PrivateTensor
from stensorflow.homo_enc.homo_mat_mul import matmul_homo_offline, matmul_homo_online, matmul_homo
from stensorflow.homo_enc.homo_mul import mul_homo, mul_homo_offline, mul_homo_online


def add(x: Union[PrivateTensorBase, SharedPairBase, int, float, np.ndarray, tf.Tensor],
        y: Union[PrivateTensorBase, SharedPairBase, int, float, np.ndarray, tf.Tensor]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_pair = isinstance(x, SharedPairBase)
    x_is_public = isinstance(x, (np.ndarray, tf.Tensor, float, int))
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_pair = isinstance(y, SharedPairBase)
    y_is_public = isinstance(y, (np.ndarray, tf.Tensor, float, int))

    if x_is_private and y_is_private:
        if x.owner == y.owner:
            return PrivateTensorBase.__add__(x, y)
        else:
            fixedpoint = min(x.fixedpoint, y.fixedpoint)
            altered_x = x.dup_with_precision(fixedpoint)
            altered_y = y.dup_with_precision(fixedpoint)
            return SharedPairBase(ownerL=altered_x.owner, ownerR=altered_y.owner, xL=altered_x.to_SharedTensor(),
                                  xR=altered_y.to_SharedTensor(), fixedpoint=fixedpoint)
    elif x_is_private and y_is_pair:
        fixedpoint = min(x.fixedpoint, y.fixedpoint)
        altered_x = x.dup_with_precision(fixedpoint)
        altered_y = y.dup_with_precision(fixedpoint)
        if x.owner == y.ownerL:
            with tf.device(y.ownerL):
                zL = altered_x.to_SharedTensor() + altered_y.xL
            return SharedPairBase(ownerL=y.ownerL, ownerR=y.ownerR, xL=zL, xR=altered_y.xR, fixedpoint=fixedpoint)
        elif x.owner == y.ownerR:
            with tf.device(y.ownerR):
                zR = altered_x.to_SharedTensor() + altered_y.xR
            return SharedPairBase(ownerL=y.ownerL, ownerR=y.ownerR, xL=altered_y.xL, xR=zR, fixedpoint=fixedpoint)
        else:
            raise Exception("must have x.owner==y.ownerL or x.owner==y.ownerR")
    elif x_is_private and y_is_public:
        y1 = PrivateTensorBase(owner=x.owner, fixedpoint=x.fixedpoint)
        y1.load_from_tf_tensor(y)
        # return x + y1
        return add(x, y1)

    elif x_is_pair and y_is_private:
        return add(y, x)
    elif x_is_pair and y_is_pair:
        fixedpoint = min(x.fixedpoint, y.fixedpoint)
        altered_x = x.dup_with_precision(fixedpoint)
        altered_y = y.dup_with_precision(fixedpoint)
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            y = y.mirror()
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            raise Exception("must have x.ownerL=y.ownerL and x.ownerR==y.ownerR")
        with tf.device(x.ownerL):
            zL = altered_x.xL + altered_y.xL
        with tf.device(x.ownerR):
            zR = altered_x.xR + altered_y.xR
        return SharedPairBase(ownerL=y.ownerL, ownerR=y.ownerR, xL=zL, xR=zR, fixedpoint=fixedpoint)
    elif x_is_pair and y_is_public:
        y1 = PrivateTensorBase(owner=x.ownerR, fixedpoint=x.fixedpoint)
        y1.load_from_tf_tensor(y)
        return add(x, y1)
    elif x_is_public and y_is_private:
        return add(y, x)
    elif x_is_public and y_is_pair:
        return add(y, x)
    else:
        raise NotImplementedError


def sub(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    return add(x, -y)


def mul3p(x: Union[PrivateTensorBase, SharedPairBase],
          y: Union[PrivateTensorBase, SharedPairBase, int, float, np.ndarray, tf.Tensor], fixed_point=None,
          prf_flag=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    if fixed_point is not None:
        pass
    elif not hasattr(y, "fixedpoint"):
        fixed_point = x.fixedpoint
    else:
        fixed_point = max(x.fixedpoint, y.fixedpoint)

    if prf_flag is None:
        prf_flag = StfConfig.prf_flag

    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_pair = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_pair = isinstance(y, SharedPairBase)
    y_is_public = isinstance(y, (np.ndarray, tf.Tensor, float, int))

    if x_is_private and y_is_private:
        if x.owner == y.owner:
            result = PrivateTensorBase.__mul__(x, y)
        else:
            result = BM_PrivateTensor_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y),
                                                    prf_flag=False)
    elif x_is_private and y_is_pair:
        result = BM_PrivateTensor_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y), prf_flag=prf_flag)
    elif x_is_pair and y_is_private:
        result = BM_SharedPair_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y), prf_flag=prf_flag)
    elif x_is_pair and y_is_pair:
        result = BM_SharedPair_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y), prf_flag=prf_flag)
    elif (x_is_private or x_is_pair) and y_is_public:
        result = rmul(y, x)
    else:
        raise NotImplementedError
    return result.dup_with_precision(fixed_point)


def mul2p(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
          fixed_point=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    # print("matmul2p")
    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_shared = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_shared = isinstance(y, SharedPairBase)

    if not x_is_private and not y_is_private:
        raise NotImplementedError
    if StfConfig.pre_produce_flag:
        # print("StfConfig.pre_produce_flag=", StfConfig.pre_produce_flag)
        if x_is_private and y_is_private:
            x_adjoint = x.random_uniform_adjoint()
            y_adjoint = y.random_uniform_adjoint()
        elif x_is_private and y_is_shared:
            x_adjoint = x.random_uniform_adjoint()
            y_owner = y.ownerR if y.ownerL == x.owner else y.ownerL
            y_adjoint = PrivateTensorBase(owner=y_owner, fixedpoint=y.fixedpoint,
                                          inner_value=y.xL.random_uniform_adjoint().inner_value)
        elif x_is_shared and y_is_private:
            y_adjoint = y.random_uniform_adjoint()
            x_owner = x.ownerL if y.owner == x.ownerR else x.ownerR
            x_adjoint = PrivateTensorBase(owner=x_owner, fixedpoint=x.fixedpoint,
                                          inner_value=x.xL.random_uniform_adjoint().inner_value)
        else:
            x_adjoint = x.random_uniform_adjoint()
            y_adjoint = y.random_uniform_adjoint()
        z_adjoint = mul_homo_offline(x_adjoint, y_adjoint)
        if StfConfig.offline_model:
            pass
        else:
            x_adjoint.unserialize()
            y_adjoint.unserialize()
            z_adjoint.unserialize()
        result = mul_homo_online(x, y, x_adjoint, y_adjoint, z_adjoint, fixed_point=fixed_point)
    else:
        result = mul_homo(x, y, fixed_point=fixed_point)

    return result


def mul(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
        fixed_point=None, prf_flag=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    if StfConfig.parties == 2:
        return mul2p(x, y, fixed_point)
    else:
        return mul3p(x, y, fixed_point, prf_flag)


def matmul3p(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
             fixed_point=None, prf_flag=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    if fixed_point is None:
        fixed_point = max(x.fixedpoint, y.fixedpoint)
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag

    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_pair = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_pair = isinstance(y, SharedPairBase)

    if x_is_private and y_is_private:
        if x.owner == y.owner:
            result = PrivateTensorBase.__matmul__(x, y)
        else:
            result = BM_PrivateTensor_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                                    prf_flag=prf_flag)
    elif x_is_private and y_is_pair:
        result = BM_PrivateTensor_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                             prf_flag=prf_flag)
    elif x_is_pair and y_is_private:
        result = BM_SharedPair_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                             prf_flag=prf_flag)
    elif x_is_pair and y_is_pair:
        result = BM_SharedPair_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y), prf_flag=prf_flag)
    else:
        raise Exception("type exception for type(x)={}, type(y)={}".format(type(x), type(y)))
    return result.dup_with_precision(fixed_point)


def matmul2p(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
             fixed_point=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_shared = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_shared = isinstance(y, SharedPairBase)

    if not x_is_private and not y_is_private:
        raise NotImplementedError
    if StfConfig.pre_produce_flag:
        if x_is_private and y_is_private:
            x_adjoint = x.random_uniform_adjoint()
            y_adjoint = y.random_uniform_adjoint()
        elif x_is_private and y_is_shared:
            x_adjoint = x.random_uniform_adjoint()
            y_owner = y.ownerR if y.ownerL == x.owner else y.ownerL
            y_adjoint = PrivateTensorBase(owner=y_owner, fixedpoint=y.fixedpoint,
                                          inner_value=y.xL.random_uniform_adjoint().inner_value)
        elif x_is_shared and y_is_private:
            y_adjoint = y.random_uniform_adjoint()
            x_owner = x.ownerL if y.owner == x.ownerR else x.ownerR
            x_adjoint = PrivateTensorBase(owner=x_owner, fixedpoint=x.fixedpoint,
                                          inner_value=x.xL.random_uniform_adjoint().inner_value)
        else:
            x_adjoint = x.random_uniform_adjoint()
            y_adjoint = y.random_uniform_adjoint()

        z_adjoint = matmul_homo_offline(x_adjoint, y_adjoint)
        if StfConfig.offline_model:
            pass
        else:
            x_adjoint.unserialize()
            y_adjoint.unserialize()
            z_adjoint.unserialize()
        result = matmul_homo_online(x, y, x_adjoint, y_adjoint, z_adjoint, fixed_point=fixed_point)
    else:
        result = matmul_homo(x, y, fixed_point=fixed_point)

    return result


def matmul(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
           fixed_point=None, prf_flag=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    if StfConfig.parties == 2:
        return matmul2p(x, y, fixed_point)
    else:
        return matmul3p(x, y, fixed_point, prf_flag)


def rmul_matmul(x: Union[int, float, np.ndarray, tf.Tensor], y: Union[PrivateTensorBase, SharedPairBase],
                mul_func, fixed_point=None) -> Union[PrivateTensorBase, SharedPairBase]:
    if isinstance(x, int) or isinstance(x, np.ndarray) and x.dtype == np.int \
            or isinstance(x, tf.Tensor) and x.dtype in (
            tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32, tf.uint64):
        fixedpoint_x = 0
    else:
        fixedpoint_x = StfConfig.default_fixed_point
        x = tf.cast(x * (1 << fixedpoint_x), 'int64')

    if isinstance(y, PrivateTensorBase):
        with tf.device(y.owner):
            inner_value = tf.cast(mul_func(x, y.inner_value), 'int64')
            if y.module is not None:
                inner_value = inner_value % y.module
        w = PrivateTensorBase(owner=y.owner, inner_value=inner_value, module=y.module,
                              fixedpoint=fixedpoint_x + y.fixedpoint)
    elif isinstance(y, SharedPairBase):
        x = SharedTensorBase(inner_value=x, module=y.xL.module)
        with tf.device(y.ownerL):
            yL = mul_func(x, y.xL)

        with tf.device(y.ownerR):
            yR = mul_func(x, y.xR)
        w = SharedPairBase(ownerL=y.ownerL, ownerR=y.ownerR, xL=yL, xR=yR, fixedpoint=fixedpoint_x + y.fixedpoint)
    else:
        raise Exception("y must be PrivateTensorBase or SharedPairBase")
    if fixed_point is None:
        fixed_point = max(fixedpoint_x, y.fixedpoint)
        return w.dup_with_precision(new_fixedpoint=fixed_point)


def rmul(x: Union[int, float, np.ndarray, tf.Tensor], y: Union[PrivateTensorBase, SharedPairBase],
         fixed_point=None) -> Union[PrivateTensorBase, SharedPairBase]:
    return rmul_matmul(x, y, lambda a, b: a * b, fixed_point)


#
def rmatmul(x: Union[np.ndarray, tf.Tensor], y: Union[PrivateTensorBase, SharedPairBase], fixed_point=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    return rmul_matmul(x, y, lambda a, b: a @ b, fixed_point)


def truediv(x: Union[PrivateTensorBase, SharedPairBase], y: Union[int, float, np.ndarray, tf.Tensor]) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    return rmul(1.0 / y, x)
