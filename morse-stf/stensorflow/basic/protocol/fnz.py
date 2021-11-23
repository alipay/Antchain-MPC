#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : fnz
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-04 19:13
   Description : Find the First Non-Zero bit:  https://eprint.iacr.org/2021/857

"""
import tensorflow as tf
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase
from stensorflow.exception.exception import StfCondException
from stensorflow.global_var import StfConfig
from stensorflow.random.random import get_seed, gen_ZZn_from_seed, gen_ZZn
from sympy.ntheory.modular import isprime


class ElementInSemiDirectProductZZnFpxn:
    def __init__(self, p, n, shape):
        self.p = p
        self.n = n
        self.shape = shape
        self.i = SharedTensorBase(module=n, shape=shape)
        self.u = SharedTensorBase(module=p, shape=shape + [n])

    def from_SharedTensor(self, i: SharedTensorBase, u: SharedTensorBase):
        if i.module != self.n:
            raise Exception("must have i.module == self.n")
        if u.module != self.p:
            raise Exception("must have u.module == self.p")
        if i.shape != u.shape[:-1]:
            raise Exception("must have i.shape == u.shape[:-1]")
        self.i = i
        self.u = u  # LiTu

    def load_from_tf_tensor(self, i: tf.Tensor, u: tf.Tensor):
        if i.shape != u.shape[:-1]:
            raise Exception("must have i.shape == u.shape[:-1]")
        self.i.inner_value = tf.cast(i, 'int64') % self.n
        self.u.inner_value = tf.cast(u, 'int64') % self.p

    def from_uniform_distribution(self, u_shape, seed=None):
        i_shape = u_shape[:-1]
        if seed is None:
            # self.i = SharedTensorBase(inner_value=tf.random.uniform(shape=i_shape, minval=0, maxval=self.n,
            #                                                         dtype='int64', seed=seed), module=self.n)
            self.i = SharedTensorBase(inner_value=gen_ZZn(shape=i_shape, module=self.n), module=self.n)

            # self.u = SharedTensorBase(inner_value=tf.random.uniform(shape=u_shape, minval=1, maxval=self.p,
            #                                                         dtype='int64', seed=seed), module=self.p)
            self.u = SharedTensorBase(inner_value=1 + gen_ZZn(shape=u_shape, module=self.p - 1), module=self.p)
        else:
            self.i = SharedTensorBase(inner_value=gen_ZZn_from_seed(shape=i_shape, seed=seed, module=self.n),
                                      module=self.n)
            self.u = SharedTensorBase(inner_value=1 + gen_ZZn_from_seed(shape=u_shape, seed=seed, module=self.p - 1),
                                      module=self.p)

    # def from_uniform_PRF(self, seed, scope, id, u_shape):
    #     i_shape = u_shape[:-1]
    #     self.i = PRF_v0(seed=seed, scope=scope, id=id, shape=i_shape, module=self.n)
    #
    #     self.u = PRF_v0(seed=seed, scope=scope, id=id, shape=u_shape, module=self.p - 1)
    #
    #     self.u.inner_value = self.u.inner_value + 1
    #     self.u.module = self.p

    def __mul__(self, other):
        if not isinstance(other, ElementInSemiDirectProductZZnFpxn):
            raise Exception("other must be an Element_in_semi_direct_product_ZnZ_Fpxn")
        if self.n != other.n:
            raise Exception("must have self.n == other.n")
        if self.p != other.p:
            raise Exception("must have self.p == other.p")
        i = self.i + other.i
        u = self.u.rshift(other.i.inner_value) * other.u
        z = ElementInSemiDirectProductZZnFpxn(p=self.p, n=self.n, shape=self.shape)
        z.from_SharedTensor(i, u)
        # p.load_from_tf_tensor(i.inner_value, u.inner_value)
        return z

    def __invert__(self):  # L_iT_u L_{-i}T_(L_iu^{-1})=L_i T_u T_{u^-1} L_{-i}=1
        u = (self.u ** (self.p - 2)).lshift(self.i.inner_value)
        invert = ElementInSemiDirectProductZZnFpxn(p=self.p, n=self.n, shape=self.shape)
        invert.from_SharedTensor(-self.i, u)
        return invert

    # def __invert__(self):  # L_iT_u L_{-i}T_(L_iu^{-1})=L_i T_u T_{u^-1} L_{-i}=1
    #     return self

    def act(self, x: SharedTensorBase):
        if not isinstance(x, SharedTensorBase):
            raise Exception("other must be a SharedTensorBase")
        if self.n != x.shape[-1]:
            raise Exception("must have self.n == x.shape[-1]")
        if self.p != x.module:
            raise Exception("must have self.p == x.module")
        return (self.u * x).lshift(self.i.inner_value)

    def to_compress_tensor(self, dtype=tf.dtypes.int64):
        compressed_i = self.i.to_compress_tensor(dtype)
        compressed_u = self.u.to_compress_tensor(dtype)
        return (compressed_i, compressed_u)

    def decompress_from(self, compressed_i, compressed_u, shape_i, shape_u):
        self.i.decompress_from(compressed_i, shape_i)
        self.u.decompress_from(compressed_u, shape_u)


# def _G_module_recover_without_PRF(b: SharedPairBase):
#     """
#
#     :param b: in A
#     :return: (g, a) in G x A s.t ga=b
#     """
#     if b.xL.module is None:
#         raise Exception("must have x.xL.module is not None")
#     with tf.device(StfConfig.RS[0]):
#         g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
#         g.from_uniform_distribution(u_shape=b.xL.shape)
#
#         u = b.xR.random_uniform_adjoint()
#         v = g.act(u)
#         vL = v.random_uniform_adjoint()
#         vR = v - vL
#     with tf.device(b.ownerR):
#         c = b.xR - vR
#     with tf.device(b.ownerL):
#         w = (~g).act(b.xL - vL + c)
#     with tf.device(b.ownerR):
#         a = w + u
#     return (g, a)


# def _G_module_recover_without_compress(b: SharedPairBase, prf_flag=StfConfig.prf_flag):
#     """
#
#     :param b: in A
#     :return: (g, a) in G x A s.t ga=b
#     """
#     if b.xL.module is None:
#         raise Exception("must have x.xL.module is not None")
#     with tf.device(StfConfig.RS[0]):
#         g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
#         if prf_flag:
#             seed_g = get_seed()
#             seed_u = get_seed()
#             seed_v = get_seed()
#             g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
#             u = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_u)
#         else:
#             g.from_uniform_distribution(u_shape=b.xL.shape)
#             u = b.to_SharedTensor_like().random_uniform_adjoint()
#         v = g.act(u)
#         if prf_flag:
#             vL = v.random_uniform_adjoint(seed=seed_v)
#         else:
#             vL = v.random_uniform_adjoint()
#         vR = v - vL
#     with tf.device(b.ownerR):
#         c = b.xR - vR
#     with tf.device(b.ownerL):
#         if prf_flag:
#             g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
#             vL = v.random_uniform_adjoint(seed=seed_v)
#         w = (~g).act(b.xL - vL + c)
#     with tf.device(b.ownerR):
#         if prf_flag:
#             u = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_u)
#         a = w + u
#     return (g, a)
#
#
# def _G_module_recover_without_PRF_with_compress(b: SharedPairBase):
#     """
#
#     :param b: in A
#     :return: (g, a) in G x A s.t ga=b
#     """
#     if b.xL.module is None:
#         raise Exception("must have x.xL.module is not None")
#     with tf.device(StfConfig.RS[0]):
#         g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
#         g.from_uniform_distribution(u_shape=b.xL.shape)
#         u = b.xR.random_uniform_adjoint()
#         v = g.act(u)
#         vL = v.random_uniform_adjoint()
#         vR = v - vL
#         g_compress = g.to_compress_tensor()
#         u_compress = u.to_compress_tensor()
#         vL_compress = vL.to_compress_tensor()
#         vR_compress = vR.to_compress_tensor()
#     with tf.device(b.ownerR):
#         vR.decompress_from(vR_compress, vR.shape)
#         c = b.xR - vR
#         c_compress = c.to_compress_tensor()
#     with tf.device(b.ownerL):
#         g.decompress_from(g_compress[0], g_compress[1], g.i.shape, g.u.shape)
#         vL.decompress_from(vL_compress, vL.shape)
#         c.decompress_from(c_compress, c.shape)
#         w = (~g).act(b.xL - vL + c)
#         w_compress = w.to_compress_tensor()
#     with tf.device(b.ownerR):
#         w.decompress_from(w_compress, w.shape)
#         u.decompress_from(u_compress, u.shape)
#         a = w + u
#     return (g, a)


def _G_module_recover(b: SharedPairBase, prf_flag=None, compress_flag=None):
    """

    :param b: in A
    :return: (g, a) in G x A s.t ga=b
    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if b.xL.module is None:
        raise Exception("must have x.xL.module is not None")
    with tf.device(StfConfig.RS[0]):
        g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
        if prf_flag:
            seed_g = get_seed()
            seed_u = get_seed()
            seed_v = get_seed()
            g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
            u = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_u)
        else:
            g.from_uniform_distribution(u_shape=b.shape)
            u = b.to_SharedTensor_like().random_uniform_adjoint()
        v = g.act(u)
        if prf_flag:
            vL = v.random_uniform_adjoint(seed=seed_v)
        else:
            vL = v.random_uniform_adjoint()
        vR = v - vL
        if compress_flag:
            if not prf_flag:
                g_compress = g.to_compress_tensor()
                u_compress = u.to_compress_tensor()
                vL_compress = vL.to_compress_tensor()
            vR_compress = vR.to_compress_tensor()
    with tf.device(b.ownerR):
        if compress_flag:
            vR.decompress_from(vR_compress, vR.shape)
        c = b.xR - vR
        if compress_flag:
            c_compress = c.to_compress_tensor()
    with tf.device(b.ownerL):
        if prf_flag:
            g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
            vL = v.random_uniform_adjoint(seed=seed_v)
        elif compress_flag:
            g.decompress_from(g_compress[0], g_compress[1], g.i.shape, g.u.shape)
            vL.decompress_from(vL_compress, vL.shape)
        if compress_flag:
            c.decompress_from(c_compress, c.shape)
        w = (~g).act(b.xL - vL + c)
        if compress_flag:
            w_compress = w.to_compress_tensor()
    with tf.device(b.ownerR):
        if prf_flag:
            u = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_u)
        elif compress_flag:
            u.decompress_from(u_compress, u.shape)
        if compress_flag:
            w.decompress_from(w_compress, w.shape)
        a = w + u
    return (g, a)


# def _G_module_recover(b: SharedPairBase, PRF: bool = False, compress=False):
#     if not PRF and not compress:
#         return _G_module_recover_without_PRF(b)
#     elif not PRF and compress:
#         return _G_module_recover_without_PRF_with_compress(b)
#     # elif PRF and compress:
#     #     return _G_module_recover_with_PRF_with_compress(b)
#     else:
#         raise NotImplementedError

#
# def _G_module_recover_inverse_without_PRF_without_compress(b: SharedPairBase):
#     """
#
#     :param b: in A
#     :return: (g, a) in G x A s.t (~g)a=b
#     """
#     if b.xL.module is None:
#         raise Exception("must have x.xL.module is not None")
#     with tf.device(StfConfig.RS[0]):
#         g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
#         g.from_uniform_distribution(u_shape=b.shape)
#         b_adjointR = b.xR.random_uniform_adjoint()
#         b_adjointL = b.xL.random_uniform_adjoint()
#         u = g.act(b_adjointL + b_adjointR)
#     with tf.device(b.ownerR):
#         delta_bR = b.xR - b_adjointR
#     with tf.device(b.ownerL):
#         delta_bL = b.xL - b_adjointL
#         w = g.act(delta_bL + delta_bR)
#     with tf.device(b.ownerR):
#         a = w + u
#     return g, a

#
# def _G_module_recover_inverse_without_PRF_with_compress(b: SharedPairBase):
#     """
#
#     :param b: in A
#     :return: (g, a) in G x A s.t (~g)a=b
#     """
#     if b.xL.module is None:
#         raise Exception("must have x.xL.module is not None")
#     with tf.device(StfConfig.RS[0]):
#         g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
#         g.from_uniform_distribution(u_shape=b.shape)
#         b_adjointR = b.xR.random_uniform_adjoint()
#         b_adjointL = b.xL.random_uniform_adjoint()
#         u = g.act(b_adjointL + b_adjointR)
#
#         g_compress_i, g_compress_u = g.to_compress_tensor()
#         u_compress = u.to_compress_tensor()
#         b_adjointL_compress = b_adjointL.to_compress_tensor()
#         b_adjointR_compress = b_adjointR.to_compress_tensor()
#     with tf.device(b.ownerR):
#         b_adjointR.decompress_from(b_adjointR_compress, b_adjointR.shape)
#         delta_bR = b.xR - b_adjointR
#         delta_bR_compress = delta_bR.to_compress_tensor()
#     with tf.device(b.ownerL):
#         g.decompress_from(shape_i=g.i.shape, shape_u=g.u.shape, compressed_i=g_compress_i, compressed_u=g_compress_u)
#         b_adjointL.decompress_from(b_adjointL_compress, b_adjointL.shape)
#         delta_bR.decompress_from(delta_bR_compress, delta_bR.shape)
#         delta_bL = b.xL - b_adjointL
#         w = g.act(delta_bL + delta_bR)
#         w_compress = w.to_compress_tensor()
#     with tf.device(b.ownerR):
#         w.decompress_from(w_compress, w.shape)
#         u.decompress_from(u_compress, u.shape)
#         a = w + u
#     return (g, a)


def _G_module_recover_inverse(b: SharedPairBase, prf_flag=None, compress_flag=None):
    """

    :param b: in A
    :return: (g, a) in G x A s.t (~g)a=b
    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if b.xL.module is None:
        raise Exception("must have x.xL.module is not None")
    with tf.device(StfConfig.RS[0]):
        g = ElementInSemiDirectProductZZnFpxn(p=b.xL.module, n=b.shape[-1], shape=b.shape[0:-1])
        if prf_flag:
            seed_g = get_seed()
            seed_bL = get_seed()
            seed_bR = get_seed()
            g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
            b_adjointL = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_bL)
            b_adjointR = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_bR)
        else:
            g.from_uniform_distribution(u_shape=b.shape)
            b_adjointR = b.to_SharedTensor_like().random_uniform_adjoint()
            b_adjointL = b.to_SharedTensor_like().random_uniform_adjoint()

        u = g.act(b_adjointL + b_adjointR)
        if compress_flag:
            if not prf_flag:
                g_compress_i, g_compress_u = g.to_compress_tensor()
                b_adjointL_compress = b_adjointL.to_compress_tensor()
                b_adjointR_compress = b_adjointR.to_compress_tensor()
            u_compress = u.to_compress_tensor()

    with tf.device(b.ownerR):
        if prf_flag:
            b_adjointR = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_bR)
        elif compress_flag:
            b_adjointR.decompress_from(b_adjointR_compress, b_adjointR.shape)
        delta_bR = b.xR - b_adjointR
        if compress_flag:
            delta_bR_compress = delta_bR.to_compress_tensor()
    with tf.device(b.ownerL):
        if prf_flag:
            g.from_uniform_distribution(u_shape=b.shape, seed=seed_g)
            b_adjointL = b.to_SharedTensor_like().random_uniform_adjoint(seed=seed_bL)
        elif compress_flag:
            g.decompress_from(shape_i=g.i.shape, shape_u=g.u.shape, compressed_i=g_compress_i,
                              compressed_u=g_compress_u)
            b_adjointL.decompress_from(b_adjointL_compress, b_adjointL.shape)
        if compress_flag:
            delta_bR.decompress_from(delta_bR_compress, delta_bR.shape)
        delta_bL = b.xL - b_adjointL
        w = g.act(delta_bL + delta_bR)
        if compress_flag:
            w_compress = w.to_compress_tensor()
    with tf.device(b.ownerR):
        if compress_flag:
            w.decompress_from(w_compress, w.shape)
            u.decompress_from(u_compress, u.shape)
        a = w + u
    return (g, a)


def fnz_v1(x: SharedPairBase, prf_flag=None, compress_flag=None):
    """

    :param b: SharedPair with module is a prime number p and 0-1 value,
    there exists at least one bit i such that x[i]==1
    and p>=b.shape[-1]+2
    :return: SharedPair, the first non-zero bit along axis=-1
    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    p = x.xL.module
    if not isprime(p):
        raise StfCondException("p be a prime number", "p={}".format(p))
    if p < x.shape[-1] + 2:
        raise Exception("must have p>=x.shape[-1]+2")

    x_p = x.cumulative_sum(axis=-1)
    y = x_p - 2 * x + x.ones_like()

    g, a = _G_module_recover(y, prf_flag=prf_flag, compress_flag=compress_flag)

    with tf.device(x.ownerL):
        i = g.i
    with tf.device(x.ownerR):
        equel0_onehot = tf.cast(tf.equal(a.inner_value, 0), 'int32')

        _range = range(equel0_onehot.shape.as_list()[-1])
        j = tf.tensordot(equel0_onehot, _range, axes=[-1, 0])
        j = tf.cast(j, 'int64')
        j = SharedTensorBase(inner_value=j, module=len(_range))
    return SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=-i, xR=j, fixedpoint=0)


def fnz_v2(x: SharedPairBase, prf_flag=None, compress_flag=None):
    """

    :param b: SharedPair with module is a prime number p and 0-1 value,
    there exists at least one bit i such that x[i]==1
    and p>=b.shape[-1]+2
    :return: SharedPair, the first non-zero bit along axis=-1
    """
    p = x.xL.module
    if not isprime(p):
        raise Exception("x.xL.module must be an integer")
    if p < x.shape[-1] + 2:
        raise Exception("must have p>=x.shape[-1]+2")
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    x_p = x.cumulative_sum(axis=-1)
    y = x_p - 2 * x + x.ones_like()

    g, a = _G_module_recover(y, prf_flag=prf_flag, compress_flag=compress_flag)

    with tf.device(x.ownerL):
        i = g.i
    with tf.device(x.ownerR):
        _range = range(a.shape[-1])
        j = tf.reduce_sum(tf.where(tf.equal(a.inner_value, 0), _range, [0]), axis=-1)
        j = tf.cast(j, 'int64')
        j = SharedTensorBase(inner_value=j, module=len(_range))
    return SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=-i, xR=j, fixedpoint=0)


def fnz_v3(x: SharedPairBase, prf_flag=None, compress_flag=None):
    """
    利用 _G_module_recover_inverse 进行 fnz
    :param b: SharedPair with module is a prime number p and 0-1 value,
    there exists at least one bit i such that x[i]==1
    and p>=b.shape[-1]+2
    :return: SharedPair, the first non-zero bit along axis=-1
    """
    p = x.xL.module
    if not isprime(p):
        raise Exception("x.xL.module must be an integer")
    if p < x.shape[-1] + 2:
        raise Exception("must have p>=x.shape[-1]+2")
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    x_p = x.cumulative_sum(axis=-1)
    y = x_p - 2 * x + x.ones_like()
    g, a = _G_module_recover_inverse(y, prf_flag=prf_flag, compress_flag=compress_flag)  # gy=a
    with tf.device(x.ownerL):
        i = g.i
    with tf.device(x.ownerR):
        j = tf.argmin(input=a.inner_value, axis=-1)
        j = tf.cast(j, 'int64')
        j = SharedTensorBase(inner_value=j, module=y.shape[-1])
    return SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=i, xR=j, fixedpoint=0)
