#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : pair.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-09 15:49
   Description : description what the main function of this file
"""
from stensorflow.basic.operator.arithmetic import add, sub, mul, matmul, rmul, rmatmul, truediv
from stensorflow.basic.operator.order import greater, geq, less, leq
from stensorflow.basic.basic_class.share import SharedVariable
from stensorflow.basic.basic_class.base import SharedPairBase, SharedTensorBase, PrivateTensorBase, get_device
import tensorflow as tf
from stensorflow.exception.exception import StfException
from stensorflow.global_var import StfConfig
import numpy as np


class SharedPair(SharedPairBase):

    def __init__(self, ownerL, ownerR, xL: SharedTensorBase = None, xR: SharedTensorBase = None,
                 fixedpoint: int = StfConfig.default_fixed_point,
                 shape=None, op_map={}):
        super(SharedPair, self).__init__(ownerL=ownerL, ownerR=ownerR, xL=xL, xR=xR, fixedpoint=fixedpoint, shape=shape)
        self.op_map = {"add": add, "sub": sub, "mul": mul, "matmul": matmul, "rmul": rmul,
                       "rmatmul": rmatmul,
                       "truediv": truediv, "gt": greater, "ge": geq, "lt": less, "le": leq}
        self.op_map.update(op_map)

    def __repr__(self) -> str:
        return 'SharedPair(ownerL={},ownerR={}, fixedpoint={}, module={}, shape={}, xL={}, xR={})'.format(
            self.ownerL.to_string(), self.ownerR.to_string(), self.fixedpoint, self.xL.module, self.xL.shape, self.xL,
            self.xR)

    def __getitem__(self, index):
        xL = self.xL[index]
        xR = self.xR[index]
        z = SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR,
                       fixedpoint=self.fixedpoint, op_map=self.op_map)
        return z


    @classmethod
    def from_SharedPairBase(cls, x: SharedPairBase, op_map={}):
        return SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=x.xL, xR=x.xR, fixedpoint=x.fixedpoint, op_map=op_map)

    def _align_owner(self, other):
        "See SharedPairBase._align_owner."
        result = super(SharedPair, self)._align_owner(other)
        return self.from_SharedPairBase(result)

    def ones_like(self):
        "See SharedPairBase.ones_like."
        result = SharedPairBase.ones_like(self)
        return self.from_SharedPairBase(result, self.op_map)

    def lower_bound_like(self):
        "Return a SharedPair of value=StfConfig.lower_bd_int64"
        with tf.device(self.ownerL):
            xL = (StfConfig.lower_bd_int64 + 1) * self.xL.ones_like()
        with tf.device(self.ownerR):
            xR = - self.xR.ones_like()
        x = SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint,
                       op_map=self.op_map)
        return x

    def upper_bound_like(self):
        with tf.device(self.ownerL):
            xL = (StfConfig.upper_bd_int64 + 1) * self.xL.ones_like()
        with tf.device(self.ownerR):
            xR = - self.xR.ones_like()
        x = SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint,
                       op_map=self.op_map)
        return x

    def zeros_like(self):
        y = SharedPairBase.zeros_like(self)
        return SharedPair.from_SharedPairBase(y, self.op_map)

    def ones_like(self):
        y = SharedPairBase.ones_like(self)
        return SharedPair.from_SharedPairBase(y, self.op_map)

    def dup_with_precision(self, new_fixedpoint):
        result = SharedPairBase.dup_with_precision(self, new_fixedpoint)
        return self.from_SharedPairBase(result, self.op_map)

    def split(self, size_splits, axis: int = 0, num=None):
        with tf.device(self.ownerL):
            xLs = self.xL.split(size_splits=size_splits, axis=axis, num=num)
        with tf.device(self.ownerR):
            xRs = self.xR.split(size_splits=size_splits, axis=axis, num=num)
        xs = []
        for xL, xR in zip(xLs, xRs):
            xs += [SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint,
                             op_map=self.op_map)]

        return tuple(xs)

    def concat(self, other, axis):
        if isinstance(other, SharedPairBase):
            result = super(SharedPair, self).concat(other, axis)
            return SharedPair.from_SharedPairBase(result, op_map=self.op_map)
        elif isinstance(other, PrivateTensorBase):  # can not import PrivateTensor
            if other.owner == self.ownerL:
                oppo_owner = self.ownerR
            elif other.owner == self.ownerR:
                oppo_owner = self.ownerL
            else:
                raise Exception("must have other.owner in {self.ownerL, self.ownerR}")
            other = other.to_SharedPair(other_owner=oppo_owner)
            return self.concat(other, axis)
        else:
            raise Exception("other must be SharedPairBase or PrivateTensor.")

    def stack(self, other, axis):
        if isinstance(other, SharedPairBase):
            result = super(SharedPair, self).stack(other, axis)
            return SharedPair.from_SharedPairBase(result, op_map=self.op_map)
        elif isinstance(other, PrivateTensorBase):
            if other.owner == self.ownerL:
                oppo_owner = self.ownerR
            elif other.owner == self.ownerR:
                oppo_owner = self.ownerL
            else:
                raise Exception("must have other.owner in {self.ownerL, self.ownerR}")
            other = other.to_SharedPair(other_owner=oppo_owner)
            return self.stack(other, axis)
        else:
            raise Exception("other must be SharedPairBase or PrivateTensor.")

    def transpose(self):
        with tf.device(self.ownerL):
            xL_t = self.xL.transpose()
            xR_t = self.xR.transpose()
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL_t, xR=xR_t, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)

    def to_SharedPair(self, ownerL, ownerR):
        ownerL = get_device(ownerL)
        ownerR = get_device(ownerR)
        if ownerL == self.ownerL and ownerR == self.ownerR:
            return self
        elif ownerL == self.ownerR and ownerR == self.ownerL:
            return self.mirror()
        else:
            raise StfException("must have Set(ownerL, ownerR)==Set(self.ownerL, self.ownerR), "
                               "but ownerL={}, ownerR={}, self.ownerL={}, self.ownerR={}"
                               .format(ownerL, ownerR, self.ownerL, self.ownerR))

    def reduce_mean(self, axis, keepdims=False):
        with tf.device(self.ownerL):
            xL = self.xL.reduce_mean(axis=axis, keepdims=keepdims)
        with tf.device(self.ownerR):
            xR = self.xR.reduce_mean(axis=axis, keepdims=keepdims)
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR,
                          fixedpoint=self.fixedpoint, op_map=self.op_map)

    def reduce_sum(self, axis, keepdims=False):
        with tf.device(self.ownerL):
            xL = self.xL.reduce_sum(axis=axis, keepdims=keepdims)
        with tf.device(self.ownerR):
            xR = self.xR.reduce_sum(axis=axis, keepdims=keepdims)
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR,
                          fixedpoint=self.fixedpoint, op_map=self.op_map)

    def squeeze(self, axis=None):
        with tf.device(self.ownerL):
            xL = self.xL.squeeze(axis=axis)
        with tf.device(self.ownerR):
            xR = self.xR.squeeze(axis=axis)
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)

    def expend_dims(self, axis):
        with tf.device(self.ownerL):
            xL = self.xL.expand_dims(axis=axis)
        with tf.device(self.ownerR):
            xR = self.xR.expand_dims(axis=axis)
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)

    def reshape(self, shape):
        """
        See tf.reshape.
        :param shape:
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.reshape(shape)
        with tf.device(self.ownerR):
            xR = self.xR.reshape(shape)
        return SharedPair(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR,
                          fixedpoint=self.fixedpoint, op_map=self.op_map)


    def __add__(self, other):
        # return super(PrivateTensor, self).__add__(other)
        result = self.op_map['add'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __radd__(self, other):
        result = self.op_map['add'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __sub__(self, other):
        # return super(PrivateTensor, self).__sub__(other)
        result = self.op_map['sub'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __rsub__(self, other):
        result = self.op_map['sub'](other, self)
        return self.from_SharedPairBase(result, self.op_map)

    def __mul__(self, other):
        result = self.op_map['mul'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __rmul__(self, other):
        result = self.op_map['rmul'](other, self)
        return self.from_SharedPairBase(result, self.op_map)

    def __matmul__(self, other):
        result = self.op_map['matmul'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __rmatmul__(self, other):
        result = self.op_map['rmatmul'](other, self)

        return self.from_SharedPairBase(result, self.op_map)

    def __lt__(self, other):
        result = self.op_map['lt'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __le__(self, other):
        result = self.op_map['le'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __gt__(self, other):
        result = self.op_map['gt'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __ge__(self, other):
        result = self.op_map['ge'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __pow__(self, power: int):
        if not isinstance(power, int):
            raise Exception("power must be a non-negative integral number.")
        elif power < 0:
            raise Exception("power must be a non-negative integral number.")
        elif power == 0:
            return self.ones_like()
        elif power == 1:
            return self
        elif power == 2:
            return (self * self).dup_with_precision(new_fixedpoint=self.fixedpoint)
        else:  # power > 2:
            power_helf = power // 2
            y_power_helf = self ** power_helf
            y = y_power_helf * y_power_helf
            if power % 2 == 0:
                return y.dup_with_precision(new_fixedpoint=self.fixedpoint)
            else:
                return (y * self).dup_with_precision(new_fixedpoint=self.fixedpoint)

    def __truediv__(self, other):
        result = self.op_map['truediv'](self, other)
        return self.from_SharedPairBase(result, self.op_map)

    def __invert__(self):
        """
         only true for self>0
        :return:
        """
        x = 0.00005
        for _ in range(StfConfig.invert_iter_num):
            x = 2 * x - x * x * self
        return x

    def load_from_numpy(self, x: np.ndarray):
        with tf.device(self.ownerL):
            px = PrivateTensorBase(owner=self.ownerL)

            px.load_from_tf_tensor(tf.constant(x))
            xL = px.inner_value + tf.ones_like(px.inner_value)
            self.xL = SharedTensorBase(inner_value=xL, module=px.module)
            self.fixedpoint = px.fixedpoint
        with tf.device(self.ownerR):
            xR = -tf.ones_like(xL)
            self.xR = SharedTensorBase(inner_value=xR, module=px.module)


class SharedVariablePair(SharedPair):

    def __init__(self, ownerL, ownerR, xL: SharedTensorBase = None, xR: SharedTensorBase = None, fixedpoint: int = 14,
                 op_map={}, shape=None):
        super(SharedVariablePair, self).__init__(ownerL=ownerL, ownerR=ownerR, fixedpoint=fixedpoint,
                                                 op_map=op_map, shape=shape)
        if xL is not None and xR is not None:
            with tf.device(ownerL):
                self.xL = SharedVariable(initial_inner_value=xL.inner_value, module=xL.module)
            with tf.device(ownerR):
                self.xR = SharedVariable(initial_inner_value=xR.inner_value, module=xR.module)

    def __repr__(self) -> str:
        return 'SharedPairVariable(ownerL={}, ownerR={}, fixedpoint={}, module={}, shape={})'.format(
            self.ownerL.to_string(), self.ownerR.to_string(), self.fixedpoint, self.xL.module, self.xL.shape)

    def assign(self, other: SharedPairBase) -> tf.Operation:

        if self.fixedpoint != other.fixedpoint:
            other = other.dup_with_precision(new_fixedpoint=self.fixedpoint)
        with tf.device(self.ownerL):
            assign_opL = self.xL.assign(other.xL)
        with tf.device(self.ownerR):
            assign_opR = self.xR.assign(other.xR)
        return tf.group(assign_opL, assign_opR)

    def load_from_tf_tensor(self, x: tf.Tensor, const=False):
        """

        :param x: tf.Tensor
        :return:
        """
        if const:
            # super(SharedVariablePair, self).from_numpy(x)
            super(SharedVariablePair, self).load_from_tf_tensor(x)
        else:
            with tf.device(self.ownerL):
                px = PrivateTensorBase(owner=self.ownerL, fixedpoint=self.fixedpoint)
                px.load_from_tf_tensor(x)
                u = px.to_SharedTensor()
                uL = u + u.ones_like()
                uR = -u.ones_like()
                self.xL = SharedVariable(initial_inner_value=uL.inner_value, module=uL.module)
                self.fixedpoint = px.fixedpoint
            with tf.device(self.ownerR):
                self.xR = SharedVariable(initial_inner_value=uR.inner_value, module=uR.module)

    def load_from_numpy(self, x: np.ndarray, const=False):
        if const:
            super(SharedVariablePair, self).load_from_numpy(x)
        else:
            with tf.device(self.ownerL):
                px = PrivateTensorBase(owner=self.ownerL)

                px.load_from_tf_tensor(tf.constant(x))
                xL = px.inner_value + tf.ones_like(px.inner_value)
                self.xL = SharedVariable(initial_inner_value=xL, module=px.module)
                self.fixedpoint = px.fixedpoint
            with tf.device(self.ownerR):
                xR = -tf.ones_like(xL)
                self.xR = SharedVariable(initial_inner_value=xR, module=px.module)
