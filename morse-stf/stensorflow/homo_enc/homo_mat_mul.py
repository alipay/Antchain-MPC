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
from stensorflow.exception.exception import StfValueException, StfEqualException, StfCondException
import tensorflow as tf
from typing import Union
from stensorflow.basic.operator.algebra import stack
from stensorflow.global_var import StfConfig
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_SharedPair, BM_PrivateTensor_PrivateTensor, \
    BM_SharedPair_SharedPair, BM_SharedPair_PrivateTensor
from stensorflow.homo_enc.homo_enc import gene_key, enc, mat_mul_vec_to_share, dec

def matmulvec_homo(mat: Union[PrivateTensorBase, SharedPairBase], x: Union[PrivateTensorBase, SharedPairBase],
                fixed_point=None) -> SharedPairBase:
    mat_is_private = isinstance(mat, PrivateTensorBase)
    x_is_private = isinstance(x, PrivateTensorBase)

    if mat_is_private and not x_is_private:
        return matmulvec_privat_pair(mat,x, fixed_point)
    elif mat_is_private and x_is_private:
        return matmulvec_privat_private(mat, x, fixed_point)
    else:
        raise NotImplementedError

def matmulvec_privat_private(mat: PrivateTensorBase, x: PrivateTensorBase, fixed_point=None) -> SharedPairBase:
    if isinstance(mat, PrivateTensorBase) and isinstance(x, PrivateTensorBase):
        pass
    else:
        raise NotImplementedError
    if mat.shape[1] != x.shape[0]:
        raise StfEqualException("mat.shape[1]", "x.shape[0]", mat.shape[1], x.shape[0])
    if mat.owner == x.owner:
        return PrivateTensorBase.__matmul__(mat, x, fixed_point)
    else:
        pass
    row_num = mat.shape[0]
    with tf.device(x.owner):
        sk, pk, gk = gene_key()
        cipher_vec_x = enc(pk, x.inner_value)
    with tf.device(mat.owner):
        plaintext_out_mat, cipher_out_vec = mat_mul_vec_to_share(pk, gk, mat.inner_value,
                                                                             cipher_vec_x)
        plaintext_out_mat = tf.reshape(plaintext_out_mat, [mat.shape[0]])
    with tf.device(x.owner):
        plaintext_out_vec = dec(sk, [row_num], cipher_out_vec)
        plaintext_out_vec = tf.reshape(plaintext_out_vec, [mat.shape[0]])

    with tf.device(mat.owner):
        xL = SharedTensorBase(inner_value=plaintext_out_mat)
    with tf.device(x.owner):
        xR = SharedTensorBase(inner_value=plaintext_out_vec)
    r = SharedPairBase(ownerL=mat.owner, ownerR=x.owner, xL=xL, xR=xR,
                       fixedpoint=mat.fixedpoint + x.fixedpoint)
    if fixed_point is None:
        return r
    else:
        return r.dup_with_precision(new_fixedpoint=fixed_point)


def matmulvec_privat_pair(mat: PrivateTensorBase, x: SharedPairBase, fixed_point=None) -> SharedPairBase:
    if isinstance(mat, PrivateTensorBase) and isinstance(x, SharedPairBase):
        pass
    else:
        raise NotImplementedError
    if mat.shape[1]!=x.shape[0]:
        raise StfEqualException("mat.shape[1]", "x.shape[0]", mat.shape[1], x.shape[0])
    row_num = mat.shape[0]
    if x.ownerL == mat.owner:
        vec_owner = x.ownerR
        vec_x = x.xR.inner_value
    elif x.ownerR == mat.owner:
        vec_owner = x.ownerL
        vec_x = x.xL.inner_value
    else:
        raise StfValueException("mat.owner", "x.ownerL or x.ownerR",  mat.owner)

    with tf.device(vec_owner):
        homo_module = StfConfig.homo_module
        sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0]))
        cipher_vec_x = homo_module.enc(pk, vec_x)
    with tf.device(mat.owner):
        plaintext_out_mat, cipher_out_vec = homo_module.mat_mul_vec_to_share(pk, gk, mat.inner_value, cipher_vec_x)
        plaintext_out_mat = tf.reshape(plaintext_out_mat, [mat.shape[0]])
    with tf.device(vec_owner):
        plaintext_out_vec = homo_module.dec(sk, [row_num], cipher_out_vec)
        plaintext_out_vec = tf.reshape(plaintext_out_vec, [mat.shape[0]])

    if x.ownerL == mat.owner:
        with tf.device(x.ownerL):
            mx_local = tf.matmul(mat.inner_value, x.xL.inner_value)
            mx_local = tf.reshape(mx_local, plaintext_out_mat.shape)
            xL = SharedTensorBase(inner_value=plaintext_out_mat+mx_local)
        with tf.device(x.ownerR):
            xR = SharedTensorBase(inner_value=plaintext_out_vec)
    elif x.ownerR == mat.owner:
        with tf.device(x.ownerL):
            xL = SharedTensorBase(inner_value=plaintext_out_vec)
        with tf.device(x.ownerR):
            mx_local = tf.matmul(mat.inner_value, x.xR.inner_value)
            mx_local = tf.reshape(mx_local, plaintext_out_mat.shape)
            xR =SharedTensorBase(inner_value=plaintext_out_mat+mx_local)
    else:
        raise StfValueException("mat.owner", "x.ownerL or x.ownerR",  mat.owner)
    r = SharedPairBase(ownerL=x.ownerL,ownerR=x.ownerR, xL=xL, xR=xR,
                       fixedpoint=mat.fixedpoint+x.fixedpoint)
    if fixed_point is None:
        return r
    else:
        return r.dup_with_precision(new_fixedpoint=fixed_point)


def matmul_homo_dim2(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase], fixed_point=None) -> SharedPairBase:
    if len(x.shape) != 2:
        raise StfValueException("len(x.shape)", 2, len(x.shape))
    if len(y.shape) != 2:
        raise StfValueException("len(y.shape)", 2, len(y.shape))
    if x.shape[1] != y.shape[0]:
        raise StfEqualException("x.shape[1]", "y.shape[0]", x.shape[1], y.shape[0])
    if y.shape[1] == 1:
        z = matmulvec_homo(x, y, fixed_point=fixed_point).reshape([-1, 1])
        # print("z=",z)
    else:
        y_columns = y.split(size_splits=y.shape[1], axis=1)
        z_list = []
        for yj in y_columns:
            z_list.append(matmulvec_homo(x, yj, fixed_point=fixed_point))
        # print("z_list[0]=", z_list[0])
        z = stack(z_list, axis=1)
        print("z=", z)
    return z


def matmul_homo(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase], fixed_point=None) -> SharedPairBase:
    if len(x.shape) == 2 and len(y.shape) == 2:
        return matmul_homo_dim2(x, y, fixed_point)
    else:
        if len(x.shape) != len(y.shape):
            raise StfEqualException("len(x.shape)", "len(y.shape)", len(x.shape), len(y.shape))
        m = x.shape[-2]
        n = x.shape[-1]
        if x.shape[-1] != y.shape[-2]:
            raise StfEqualException("x.shape[-1]", "y.shape[-2]", x.shape[-1], y.shape[-2])
        p = y.shape[-1]
        reshaped_x = x.reshape([-1, m, n])
        reshaped_y = y.reshape([-1, n, p])
        # print("reshaped_x.shape=",reshaped_x.shape)
        # print("reshaped_y.shape=", reshaped_y.shape)
        split_x = reshaped_x.split(size_splits=reshaped_x.shape[0], axis=0)
        split_y = reshaped_y.split(size_splits=reshaped_y.shape[0], axis=0)
        list_reshape_z = []
        for (xi, yi) in zip(split_x, split_y):
            list_reshape_z.append(matmul_homo_dim2(xi.squeeze(axis=0), yi.squeeze(axis=0), fixed_point=fixed_point))
        reshape_z = stack(list_reshape_z, axis=0)
        z = reshape_z.reshape(x.shape[:-2]+[m, p])
        return z

def matmul_homo_offline(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]):
    # print("matmul_homo_offline")
    x.serialize()
    y.serialize()
    if len(x.shape) == 2 and len(y.shape) == 2:
        z = matmul_homo_dim2(x, y)
    else:
        if len(x.shape) != len(y.shape):
            raise StfEqualException("len(x.shape)", "len(y.shape)", len(x.shape), len(y.shape))
        m = x.shape[-2]
        n = x.shape[-1]
        if x.shape[-1] != y.shape[-2]:
            raise StfEqualException("x.shape[-1]", "y.shape[-2]", x.shape[-1], y.shape[-2])
        p = y.shape[-1]
        reshaped_x = x.reshape([-1, m, n])
        reshaped_y = y.reshape([-1, n, p])
        # print("reshaped_x.shape=",reshaped_x.shape)
        # print("reshaped_y.shape=", reshaped_y.shape)
        split_x = reshaped_x.split(size_splits=reshaped_x.shape[0], axis=0)
        split_y = reshaped_y.split(size_splits=reshaped_y.shape[0], axis=0)
        list_reshape_z = []
        for (xi, yi) in zip(split_x, split_y):
            list_reshape_z.append(matmul_homo_dim2(xi.squeeze(axis=0), yi.squeeze(axis=0)))
        reshape_z = stack(list_reshape_z, axis=0)
        z = reshape_z.reshape(x.shape[:-2]+[m, p])
    z.serialize()
    return z


def matmul_homo_online(x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase],
                       x_adjoint: Union[PrivateTensorBase, SharedPairBase], y_adjoint: Union[PrivateTensorBase, SharedPairBase],  z_adjoint: Union[PrivateTensorBase, SharedPairBase], fixed_point=None):
    x_is_private = isinstance(x, PrivateTensorBase)
    x_is_shared = isinstance(x, SharedPairBase)
    y_is_private = isinstance(y, PrivateTensorBase)
    y_is_shared = isinstance(y, SharedPairBase)

    if x_is_private and y_is_private:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        z = BM_PrivateTensor_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                           prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(), y_adjoint=y_adjoint.to_SharedTensor(),
                                           u0=z_adjoint.xL, u1=z_adjoint.xR)
    elif x_is_private and y_is_shared:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        # print("line 481 x_adjoint=", x_adjoint)
        # print("linne 482 y_adjoint=", y_adjoint)
        # print("line 483 z_adjoint.xL=", z_adjoint.xL.inner_value)
        # print("line 484 z_adjoint.xR=", z_adjoint.xR.inner_value)
        z = BM_PrivateTensor_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                        prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(),
                                        y_adjoint=y_adjoint.to_SharedTensor(), u0=z_adjoint.xL, u1=z_adjoint.xR
                                        )
    elif x_is_shared and y_is_private:
        assert isinstance(x_adjoint, PrivateTensorBase) and isinstance(y_adjoint, PrivateTensorBase)
        z = BM_SharedPair_PrivateTensor(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                        prf_flag=False, x_adjoint=x_adjoint.to_SharedTensor(),
                                        y_adjoint=y_adjoint.to_SharedTensor(),
                                        u0=z_adjoint.xL, u1=z_adjoint.xR)
    elif x_is_shared and y_is_shared:
        assert isinstance(x_adjoint, SharedPairBase) and isinstance(y_adjoint, SharedPairBase)
        z = BM_SharedPair_SharedPair(x, y, lambda _x, _y: SharedTensorBase.__matmul__(_x, _y),
                                     prf_flag=False, xL_adjoint=x_adjoint.xL, xR_adjoin=x_adjoint.xR,
                                     yL_adjoint=y_adjoint.xL, yR_adjoint=y_adjoint.xR,
                                     uL=z_adjoint.xL, uR=z_adjoint.xR)
    else:
        raise StfCondException("x, y are PrivateTensor or SharedPair", "x is, y is".format(type(x), type(y)))
    return z
