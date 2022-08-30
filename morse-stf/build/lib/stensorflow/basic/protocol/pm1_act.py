#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : pm1_act
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-12 14:43
   Description : (t, x) -> (-1)^tx : https://eprint.iacr.org/2021/857
"""

from stensorflow.basic.basic_class.pair import SharedPairBase
from stensorflow.basic.basic_class.share import SharedTensor, SharedTensorBase
from stensorflow.basic.basic_class.base import PrivateTensorBase
from stensorflow.random.random import get_seed, gen_rint64, gen_rint64_from_seed
from typing import Union
from stensorflow.exception.exception import StfCondException, StfEqualException, StfValueException, StfNoneException
from stensorflow.global_var import StfConfig
import tensorflow as tf


class SharedTensorInt65:
    def __init__(self, x_high: tf.Tensor = None, x_low: tf.Tensor = None, shape=None):
        if x_high is not None and x_low is not None:
            if x_high.dtype != tf.dtypes.int64 or x_low.dtype != tf.dtypes.bool:
                raise StfCondException("x_high.dtype == tf.dtypes.int64 and x_low.dtype != tf.dtypes.bool",
                                       "x_high.dtype ={}, x_low.dtype={}".format(x_high.dtype, x_low.dtype))
            if x_high.shape != x_low.shape:
                raise StfEqualException("x_high.shape", "x_low.shape", x_high.shape, x_low.shape)
            self.x_high = x_high
            self.x_low = x_low
            self.shape = x_high.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise StfNoneException("x_high and x_low or shape")

    @property
    def device(self):
        if self.x_high.device == self.x_low.device:
            return self.x_high.device
        else:
            return [self.x_high.device, self.x_low.device]

    def identity(self):
        x_high = tf.identity(self.x_high)
        x_low = tf.identity(self.x_low)
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    @classmethod
    def load_from_SharedTensor(cls, x: SharedTensorBase):
        x_low = tf.cast(x.inner_value % 2, 'bool')
        x_high = x.inner_value // 2
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    @classmethod
    def load_from_PrivateTensor(cls, x: PrivateTensorBase):
        x_low = tf.cast(x.inner_value % 2, 'bool')
        x_high = x.inner_value // 2
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    def to_SharedTensor(self):
        return SharedTensor(inner_value=self.x_high * 2 + tf.cast(self.x_low, 'int64'))

    def zeros_like(self):
        x_low = tf.logical_and(self.x_low, False)
        x_high = tf.zeros_like(self.x_high)
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    def __add__(self, other):
        x_low = tf.logical_or(tf.logical_and(self.x_low, tf.logical_not(other.x_low)),
                              tf.logical_and(tf.logical_not(self.x_low), other.x_low))
        carry = tf.logical_and(self.x_low, other.x_low)
        x_high = self.x_high + other.x_high + tf.cast(carry, 'int64')
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    def __neg__(self):
        x_low = self.x_low
        x_high = -self.x_high - tf.cast(x_low, 'int64')
        return SharedTensorInt65(x_high=x_high, x_low=x_low)

    def __sub__(self, other):
        return self + (-other)

    def random_uniform_adjoint(self, seed=None):
        """

        :param seed: seed for generate random number
        :return:     generate  pseudo-random SharedTensorInt65 with same shape and
                    module with self
        """
        if seed is not None:
            x = gen_rint64_from_seed(shape=[2] + self.shape, seed=seed)
        else:
            x = gen_rint64([2] + self.shape)
        x_high = x[0, ...]
        x_low = tf.cast(x[1, ...] % 2, 'bool')
        adjoint = SharedTensorInt65(x_high=x_high, x_low=x_low)
        return adjoint


class SharedPairInt65:
    def __init__(self, ownerL, ownerR, xL: SharedTensorInt65 = None, xR: SharedTensorInt65 = None,
                 fixedpoint=StfConfig.default_fixed_point):
        self.ownerL = ownerL
        self.ownerR = ownerR
        if xL is not None and xR is not None:
            with tf.device(ownerL):
                if xL.device == ownerL.to_string():
                    self.xL = xL
                else:
                    self.xL = xL.identity()
                self.shape = self.xL.shape
            with tf.device(ownerR):
                if xR.device == ownerR.to_string():
                    self.xR = xR
                else:
                    self.xR = xR.identity()
                self.ownerR = ownerR

        else:
            self.xL = None
            self.xR = None
            self.shape = None
        self.fixedpoint = fixedpoint

    @classmethod
    def from_SharedPair(cls, x: SharedPairBase):
        ownerL = x.ownerL
        ownerR = x.ownerR
        with tf.device(ownerL):
            xL = SharedTensorInt65.load_from_SharedTensor(x.xL)
        with tf.device(ownerR):
            xR = SharedTensorInt65.load_from_SharedTensor(x.xR)
        return SharedPairInt65(ownerL, ownerR, xL, xR, x.fixedpoint)

    @classmethod
    def load_from_SharedTensorInt65(cls, x: SharedTensorInt65, owner, other_owner, fixedpoint=0):
        with tf.device(owner):
            xL = x.identity()
        with tf.device(other_owner):
            xR = x.zeros_like()
        return SharedPairInt65(owner, other_owner, xL, xR, fixedpoint)

    def to_SharedPair(self):
        with tf.device(self.ownerL):
            xL = self.xL.to_SharedTensor()
        with tf.device(self.ownerR):
            xR = self.xR.to_SharedTensor()
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def half_of_even(self) -> SharedPairBase:
        with tf.device(self.ownerL):
            xL = SharedTensorBase(inner_value=self.xL.x_high)
        with tf.device(self.ownerR):
            xR = SharedTensorBase(inner_value=self.xR.x_high + tf.cast(self.xR.x_low, 'int64'))
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR,
                              xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def to_tf_tensor(self, owner=None):
        z = self.to_SharedPair()
        return z.to_tf_tensor(owner)

    def mirror(self):
        ownerR = self.ownerL
        ownerL = self.ownerR
        xR = self.xL
        xL = self.xR
        return SharedPairInt65(ownerL, ownerR, xL, xR, self.fixedpoint)

    def __add__(self, other):
        if self.fixedpoint != other.fixedpoint:
            raise StfEqualException("self.fixedpoint", "other.fixedpoint", self.fixedpoint, other.fixedpoint)
        if self.ownerL == other.ownerR and self.ownerR == other.ownerL:
            other = other.mirror()
        with tf.device(self.ownerL):
            xL = self.xL + other.xL
        with tf.device(self.ownerR):
            xR = self.xR + other.xR
        return SharedPairInt65(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def __neg__(self):
        return SharedPairInt65(ownerL=self.ownerL, ownerR=self.ownerR, xL=-self.xL, xR=-self.xR,
                               fixedpoint=self.fixedpoint)

    def __sub__(self, other):
        return self + (-other)

    def to_SharedTensor_like(self):
        return SharedTensorInt65(shape=self.shape)


def _pm1_act(t: SharedTensor, x: Union[SharedTensor, SharedTensorInt65]) -> Union[SharedTensor, SharedTensorInt65]:
    # (t, x) -> (-1)^tx
    if t.module != 2:
        raise StfValueException("t.module", 2, t.module)
    if t.shape != x.shape:
        raise StfEqualException("t.shape", "x.shape", t.shape, x.shape)
    if isinstance(x, SharedTensorBase):
        inner_value = (x.inner_value - t.inner_value * 2 * x.inner_value)
        return SharedTensor(inner_value=inner_value, module=x.module)
    elif isinstance(x, SharedTensorInt65):
        x_low = x.x_low
        x_high = (1 - 2 * t.inner_value) * x.x_high - t.inner_value * tf.cast(x.x_low, 'int64')
        return SharedTensorInt65(x_high=x_high, x_low=x_low)


def pm1_act(t: SharedTensor, x: Union[SharedTensor, SharedTensorInt65], t_owner, x_owner, RS_owner,
            prf_flag=None, compress_flag=None) \
        -> Union[SharedPairBase, SharedPairInt65]:
    # Step 1.  generate t_adjoint, x_adjoint, b_t, b_x, s.t  (-1)^t_adjoint x_adjoint = b_t + b_x
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
        b = _pm1_act(t_adjoint, x_adjoint)
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

    # Step4. compute  y_t:=(-1)^t(x-x_adjoint)+(-1)^delta_t b_t
    with tf.device(t_owner):
        y_t = _pm1_act(t, delta_x) + _pm1_act(delta_t, b_t)

    # Step 5. compute y_x:=(-1)^delta_t + b_x
    with tf.device(x_owner):
        if compress_flag:
            delta_t.decompress_from(delta_t_compress)
        y_x = _pm1_act(delta_t, b_x)
    if isinstance(x, SharedTensor):
        return SharedPairBase(ownerL=t_owner, ownerR=x_owner, xL=y_t, xR=y_x, fixedpoint=0)
    elif isinstance(x, SharedTensorInt65):
        return SharedPairInt65(ownerL=t_owner, ownerR=x_owner, xL=y_t, xR=y_x, fixedpoint=0)


def pm1_pair_act(t: SharedPairBase, x: Union[SharedTensor, SharedTensorInt65], x_owner, RS_owner,
                 prf_flag=None, compress_flag=None) \
        -> Union[SharedPairBase, SharedPairInt65]:
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
        tR_act_x = _pm1_act(t.xR, x)
    return pm1_act(t.xL, tR_act_x, t_owner=t.ownerL, x_owner=x_owner, RS_owner=RS_owner,
                   prf_flag=prf_flag, compress_flag=compress_flag)


def pm1_pair_act_pair(t: SharedPairBase, x: Union[SharedPairBase, SharedPairInt65], RS_owner, prf_flag=None,
                      compress_flag=None) -> Union[SharedPairBase, SharedPairInt65]:
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
        y = _pm1_act(tL_adjoint + tR_adjoint, xL_adjoint + xR_adjoint)
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
            uL = _pm1_act(t.xL, zR) + _pm1_act(delta_tL + delta_tR_new, yL)
        else:
            uL = _pm1_act(t.xL, zR) + _pm1_act(delta_tL + delta_tR, yL)
    with tf.device(t.ownerR):
        if compress_flag:
            delta_tL_new = delta_tL.decompress_from_to_new(delta_tL_compressed)
            uR = _pm1_act(t.xR, zL) + _pm1_act(delta_tL_new + delta_tR, yR)
        else:
            uR = _pm1_act(t.xR, zL) + _pm1_act(delta_tL + delta_tR, yR)
    if isinstance(x, SharedPairBase):
        return SharedPairBase(ownerL=t.ownerL, ownerR=t.ownerR, xL=uL, xR=uR, fixedpoint=x.fixedpoint)
    elif isinstance(x, SharedPairInt65):
        return SharedPairInt65(ownerL=t.ownerL, ownerR=t.ownerR, xL=uL, xR=uR, fixedpoint=x.fixedpoint)
