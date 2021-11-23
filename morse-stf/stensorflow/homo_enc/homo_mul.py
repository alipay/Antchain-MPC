#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : homo_mat_mul
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/9/30 下午5:31
   Description : description what the main function of this file
"""


from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from stensorflow.exception.exception import StfEqualException, StfCondException
import tensorflow as tf
from typing import Union
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_SharedPair, BM_PrivateTensor_PrivateTensor, \
    BM_SharedPair_SharedPair, BM_SharedPair_PrivateTensor
from stensorflow.homo_enc.homo_enc import gene_key, enc, vec_mul_vec_to_share, dec



def vecmulvec_privat_private(x: PrivateTensorBase, y: PrivateTensorBase, fixed_point=None) -> SharedPairBase:
    if isinstance(x, PrivateTensorBase) and isinstance(y, PrivateTensorBase):
        pass
    else:
        raise NotImplementedError
    if x.shape != y.shape:
        raise StfEqualException("x.shape", "y.shape", x.shape, y.shape)
    if x.owner == y.owner:
        return PrivateTensorBase.__mul__(x, y, fixed_point)
    else:
        pass
    vec_len = x.shape[0]
    with tf.device(x.owner):
        sk, pk, gk = gene_key()
        cipher_vec_x = enc(pk, x.inner_value)
    with tf.device(y.owner):
        cipher_out_vec, share_out_vec = vec_mul_vec_to_share(pk, y.inner_value, cipher_vec_x)

        xR = SharedTensorBase(inner_value=share_out_vec)
    with tf.device(x.owner):
        plaintext_out_vec = dec(sk, [vec_len], cipher_out_vec)
        xL = SharedTensorBase(inner_value=plaintext_out_vec)

    r = SharedPairBase(ownerL=x.owner, ownerR=y.owner, xL=xL, xR=xR,
                       fixedpoint=x.fixedpoint + y.fixedpoint)
    if fixed_point is None:
        return r
    else:
        return r.dup_with_precision(new_fixedpoint=fixed_point)


def mul_homo_privat_private(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase], fixed_point=None) -> SharedPairBase:
    if len(x.shape) == 1 and len(y.shape) == 1:
        return vecmulvec_privat_private(x, y, fixed_point)
    else:
        reshaped_x = x.reshape([-1])
        reshaped_y = y.reshape([-1])
        z = vecmulvec_privat_private(reshaped_x, reshaped_y, fixed_point=fixed_point)
        return z.reshape(x.shape)



def mul_homo(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase], fixed_point=None) -> SharedPairBase:
    y_is_private = isinstance(y, PrivateTensorBase)
    x_is_private = isinstance(x, PrivateTensorBase)

    if y_is_private and x_is_private:
        return mul_homo_privat_private(x, y, fixed_point)
    else:
        raise NotImplementedError

def mul_homo_offline(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]):
    print("matmul_homo_offline")
    x.serialize()
    y.serialize()
    z = mul_homo(x, y)
    z.serialize()
    return z


def mul_homo_online(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
                       x_adjoint: Union[PrivateTensorBase, SharedPairBase], y_adjoint: Union[PrivateTensorBase, SharedPairBase],  z_adjoint: Union[PrivateTensorBase, SharedPairBase], fixed_point=None):
    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_shared = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_shared = isinstance(y, SharedPairBase)

    if x_is_private and y_is_private:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        z = BM_PrivateTensor_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y),
                                           prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(), y_adjoint=y_adjoint.to_SharedTensor(),
                                           u0=z_adjoint.xL, u1=z_adjoint.xR)
    elif x_is_private and y_is_shared:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        z = BM_PrivateTensor_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y),
                                        prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(),
                                        y_adjoint=y_adjoint.to_SharedTensor(), u0=z_adjoint.xL, u1=z_adjoint.xR
                                        )
    elif x_is_shared and y_is_private:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        z = BM_SharedPair_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y),
                                        prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(),
                                        y_adjoint=y_adjoint.to_SharedTensor(),
                                        u0=z_adjoint.xL, u1=z_adjoint.xR)
    elif x_is_shared and y_is_shared:
        assert isinstance(x_adjoint, SharedPairBase) and isinstance(y_adjoint, SharedPairBase)
        z = BM_SharedPair_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__mul__(_x, _y),
                                     prf_flag=False, xL_adjoint=x_adjoint.xL, xR_adjoin=x_adjoint.xR,
                                     yL_adjoint=y_adjoint.xL, yR_adjoint=y_adjoint.xR,
                                     uL=z_adjoint.xL, uR=z_adjoint.xR)
    else:
        raise StfCondException("x, y are PrivateTensor or SharedPair", "x is, y is".format(type(x), type(y)))
    return z



