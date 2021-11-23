#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : bitwise
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/2/22 下午4:11
   Description : description what the main function of this file
"""
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase, get_device
import tensorflow as tf
from stensorflow.global_var import StfConfig
from typing import Union


class SharedTensorBitwise:
    def __init__(self, inner_value, shape=None):
        """

        :param inner_value: A tf.Tensor of dtype = tf.int64
        :param shape:
        """
        self.inner_value = inner_value
        if shape is not None:
            self.shape = shape
        elif inner_value is not None:
            self.shape = inner_value.shape
        else:
            raise Exception("must have inner_value is not None or shape is not None")

    @classmethod
    def from_SharedTensor(cls, x: Union[SharedTensorBase, tf.Tensor]):
        if isinstance(x, SharedTensorBase):
            return SharedTensorBitwise(inner_value=x.inner_value)
        elif isinstance(x, tf.Tensor):
            return SharedTensorBitwise(inner_value=x)

    def to_SharedTensor(self):
        return SharedTensorBase(inner_value=self.inner_value)

    def __repr__(self) -> str:
        return 'SharedTensorBitwise(inner_value={})'.format(self.inner_value)

    def __add__(self, other):
        inner_value = tf.bitwise.bitwise_xor(self.inner_value, other.inner_value)
        return SharedTensorBitwise(inner_value=inner_value)

    def __sub__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        inner_value = tf.bitwise.bitwise_and(self.inner_value, other.inner_value)
        return SharedTensorBitwise(inner_value=inner_value)

    def random_uniform_adjoint(self, seed=None):
        x = self.to_SharedTensor()
        y = x.random_uniform_adjoint(seed=seed)
        return self.from_SharedTensor(y)

    def to_PrivateTensorBitwise(self, owner):
        with tf.device(get_device(owner)):
            return PrivateTensorBitwise(owner=owner, inner_value=self.inner_value)


class PrivateTensorBitwise:
    def __init__(self, owner, inner_value: SharedTensorBitwise, shape=None):
        """

        :param owner:  str: "L" or "R" or a python.framework.device_spec.DeviceSpecV2 object
        :param inner_value: tf.int64
        :param shape:
        """
        self.owner = get_device(owner)
        with tf.device(get_device(owner)):
            self.inner_value = inner_value
        if shape is not None:
            self.shape = shape
        elif inner_value is not None:
            self.shape = inner_value.shape
        else:
            raise Exception("must have inner_value is not None or shape is not None")

    @classmethod
    def from_PrivateTensorBase(cls, x: PrivateTensorBase):
        with tf.device(x.owner):
            inner_value = SharedTensorBitwise.from_SharedTensor(x.inner_value)
        return PrivateTensorBitwise(owner=x.owner, inner_value=inner_value)

    def to_SharedTensorBitwise(self):
        return self.inner_value

    def __repr__(self) -> str:
        return 'PrivateTensorBitwise(owner={}, inner_value={})'.format(self.owner, self.inner_value)

    def __add__(self, other):
        if isinstance(other, PrivateTensorBitwise):
            if other.owner == self.owner:
                with tf.device(get_device(self.owner)):
                    inner_value = self.inner_value + other.inner_value
                return PrivateTensorBitwise(owner=self.owner, inner_value=inner_value)
            else:
                return SharedPairBitwise(ownerL=self.owner, ownerR=other.owner, xL=self.inner_value,
                                         xR=other.inner_value)
        elif isinstance(other, SharedPairBitwise):
            if other.ownerL == self.owner:
                with tf.device(get_device(self.owner)):
                    xL = self.inner_value + other.xL
                return SharedPairBitwise(ownerL=other.ownerL, ownerR=other.ownerR, xL=xL, xR=other.xR)
            elif other.ownerR == self.owner:
                with tf.device(get_device(self.owner)):
                    xR = self.inner_value + other.xR
                return SharedPairBitwise(ownerL=other.ownerL, ownerR=other.ownerR, xL=other.xL, xR=xR)
            else:
                raise Exception("must have other.ownerL == self.owner or other.ownerR == self.owner")
        else:
            raise Exception("must have isinstance(other, PrivateTensorBitwise) or isinstance(other, SharedPairBitwise)")

    def random_uniform_adjoint(self, seed=None):
        x = self.to_SharedTensorBitwise()
        y = x.random_uniform_adjoint(seed=seed)
        return y

    def __sub__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, PrivateTensorBitwise):
            if other.owner == self.owner:
                with tf.device(get_device(self.owner)):
                    inner_value = tf.bitwise.bitwise_and(self.inner_value, other.inner_value)
                    return PrivateTensorBitwise(owner=self.owner, inner_value=inner_value)
            else:
                with tf.device(StfConfig.RS[0]):
                    self_adjoint = self.random_uniform_adjoint()
                    other_adjoint = other.random_uniform_adjoint()
                    z = self_adjoint * other_adjoint
                    zL = z.random_uniform_adjoint()
                    zR = z - zL
                with tf.device(self.owner):
                    dself = self.inner_value - self_adjoint
                with tf.device(other.owner):
                    dother = other.inner_value - other_adjoint
                    uR = dself * other.inner_value + zR
                with tf.device(self.owner):
                    uL = self_adjoint * dother + zL
                return SharedPairBitwise(ownerL=self.owner, ownerR=other.owner, xL=uL, xR=uR)
        elif isinstance(other, SharedPairBitwise):
            if self.owner == other.ownerL or self.owner == other.ownerR:
                xyL = self * other.xL.to_PrivateTensorBitwise(owner=other.ownerL)
                xyR = self * other.xR.to_PrivateTensorBitwise(owner=other.ownerR)
                return xyL + xyR
            else:
                raise Exception("must have self.owner == other.ownerL or self.owner == other.ownerR")
        else:
            raise Exception("must have isinstance(other, PrivateTensorBitwise) or isinstance(other, SharedPairBitwise)")


class SharedPairBitwise:
    def __init__(self, ownerL, ownerR, xL: SharedTensorBitwise, xR: SharedTensorBitwise, shape=None):
        self.ownerL = get_device(ownerL)
        self.ownerR = get_device(ownerR)
        with tf.device(get_device(self.ownerL)):
            self.xL = xL
        with tf.device(get_device(self.ownerR)):
            self.xR = xR
        if shape is not None:
            self.shape = shape
        elif xL is not None:
            self.shape = xL.shape
        else:
            raise Exception("must have shape is not None or xL is not None")

    def __repr__(self) -> str:
        return 'SharedPairBitwise(ownerL={}, ownerR={}, xL={}, xR={})'.format(self.ownerL, self.ownerR, self.xL,
                                                                              self.xR)

    def reflex(self):
        return SharedPairBitwise(ownerL=self.ownerR, ownerR=self.ownerL, xL=self.xL, xR=self.xR)

    def to_tf_tensor(self):
        return (self.xL + self.xR).inner_value

    def to_SharedPair(self):
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=self.xL.to_SharedTensor(),
                              xR=self.xR.to_SharedTensor(), fixedpoint=0)

    @classmethod
    def from_SharedPairBase(cls, x: SharedPairBase):
        return SharedPairBitwise(ownerL=x.ownerL, ownerR=x.ownerR, xL=x.xL, xR=x.xR)

    def __add__(self, other):
        if isinstance(other, PrivateTensorBitwise):
            return other.__add__(self)
        elif isinstance(other, SharedPairBitwise):
            if self.ownerL == other.ownerL and self.ownerR == other.ownerR:
                with tf.device(self.ownerL):
                    xL = self.xL + other.xL
                with tf.device(self.ownerR):
                    xR = self.xR + other.xR
                return SharedPairBitwise(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR)
            elif self.ownerL == other.ownerR and self.ownerR == other.ownerL:
                other = other.reflex()
                return self.__add__(other)
            else:
                raise Exception("must have {self.ownerL, self.ownerR} == {other.ownerL, other.ownerR}")
        else:
            raise Exception("must have isinstance(other, PrivateTensorBitwise) or isinstance(other, SharedPairBitwise)")

    def __mul__(self, other):
        if isinstance(other, PrivateTensorBitwise):
            return other.__mul__(self)
        elif isinstance(other, SharedPairBitwise):
            if self.ownerL == other.ownerL and self.ownerR == other.ownerR:
                with tf.device(StfConfig.RS[0]):
                    xL_adjoint = self.xL.random_uniform_adjoint()
                    xR_adjoint = self.xR.random_uniform_adjoint()
                    yL_adjoint = other.xL.random_uniform_adjoint()
                    yR_adjoint = other.xR.random_uniform_adjoint()
                    z_adjoint = (xL_adjoint + xR_adjoint) * (yL_adjoint + yR_adjoint)
                    zL_adjoint = z_adjoint.random_uniform_adjoint()
                    zR_adjoint = z_adjoint - zL_adjoint
                with tf.device(self.ownerL):
                    dxL = self.xL - xL_adjoint
                    dyL = other.xL - yL_adjoint

                with tf.device(self.ownerR):
                    dxR = self.xR - xR_adjoint
                    dyR = other.xR - yR_adjoint
                    dx = dxL + dxR
                    dy = dyL + dyR
                    zR = dx * other.xR + xR_adjoint * dy + zR_adjoint

                with tf.device(self.ownerL):
                    zL = dx * other.xL + xL_adjoint * dy + zL_adjoint
                return SharedPairBitwise(ownerL=self.ownerL, ownerR=self.ownerR, xL=zL, xR=zR)
            elif self.ownerL == other.ownerR and self.ownerR == other.ownerL:
                other = other.reflex()
                return self.__mul__(other)
        else:
            raise Exception("must have isinstance(other, PrivateTensorBitwise) or isinstance(other, SharedPairBitwise)")

    def __sub__(self, other):
        return self.__add__(other)
