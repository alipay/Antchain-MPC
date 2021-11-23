#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : private
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-13 19:33
   Description : description what the main function of this file
"""

import tensorflow as tf
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase, get_device
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.operator.arithmetic import add, sub, mul, matmul, rmul, truediv
from stensorflow.basic.operator.order import greater, geq, less, leq
from stensorflow.global_var import StfConfig
from stensorflow.exception.exception import StfEqualWarning, StfCondException, StfEqualException, StfTypeException
import numpy as np


class PrivateTensor(PrivateTensorBase):
    def __init__(self, owner, fixedpoint: int = None, inner_value=None, module: int = None, op_map={}):
        super(PrivateTensor, self).__init__(owner, fixedpoint, inner_value, module)
        self.op_map = {"add": add, "sub": sub, "mul": mul, "matmul": matmul, "gt": greater, "ge": geq,
                       "lt": less, "le": leq, "rmul": rmul, "truediv": truediv}
        self.op_map.update(op_map)

    def __repr__(self) -> str:
        return 'PrivateTensor(owner={}, fixedpoint={}, module={}, shape={})'.format(self.owner.to_string(),
                                                                                    self.fixedpoint, self.module,
                                                                                    self.shape)

    @classmethod
    def from_PrivteTensorBase(cls, x: PrivateTensorBase, op_map={}):
        return PrivateTensor(owner=x.owner, inner_value=x.inner_value, fixedpoint=x.fixedpoint, module=x.module,
                             op_map=op_map)

    def transpose(self):
        with tf.device(self.owner):
            inner_value = tf.transpose(self.inner_value)
        return PrivateTensor(owner=self.owner, inner_value=inner_value, fixedpoint=self.fixedpoint, module=self.module,
                             op_map=self.op_map)

    def squeeze(self, axis):
        with tf.device(self.owner):
            inner_value = tf.squeeze(self.inner_value, axis=axis)
        return PrivateTensor(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value, module=self.module,
                             op_map=self.op_map)

    def expend_dims(self, axis):
        with tf.device(self.owner):
            inner_value = tf.expand_dims(self.inner_value, axis=axis)
        return PrivateTensor(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value, module=self.module,
                             op_map=self.op_map)

    def split(self, size_splits, axis=0, num=None):
        with tf.device(self.owner):
            inner_values = tf.split(self.inner_value, size_splits, axis, num)
            pvs = []
            for inner_value in inner_values:
                pvs += [
                    PrivateTensor(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value, module=self.module,
                                  op_map=self.op_map)]

        return tuple(pvs)

    def to_SharedPair(self, other_owner, op_map={}):
        """

        :param other_owner:
        :param op_map:
        :return:

        """
        xL = SharedTensorBase(inner_value=self.inner_value, module=self.module)
        xR = -xL.ones_like()  # use one but not zero, for div
        xL = xL + xL.ones_like()
        x = SharedPair(ownerL=self.owner, ownerR=other_owner, xL=xL, xR=xR, fixedpoint=self.fixedpoint, op_map=op_map)
        return x

    def zeros_like(self):
        with tf.device(self.owner):
            inner_value = tf.zeros_like(self.inner_value)
        x = PrivateTensor(owner=self.owner, inner_value=inner_value, module=self.module, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)
        return x

    def ones_like(self, fixedpoint=None):
        if fixedpoint is None:
            fixedpoint = self.fixedpoint
        with tf.device(self.owner):
            inner_value = (1 << fixedpoint) * tf.ones_like(self.inner_value)
        x = PrivateTensor(owner=self.owner, inner_value=inner_value, module=self.module, fixedpoint=fixedpoint,
                          op_map=self.op_map)
        return x

    def down_bound_like(self):
        with tf.device(self.owner):
            inner_value = StfConfig.lower_bd_int64 * tf.ones_like(self.inner_value, dtype='int64')
        x = PrivateTensor(owner=self.owner, inner_value=inner_value, module=self.module, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)
        return x

    def up_bound_like(self):
        with tf.device(self.owner):
            inner_value = StfConfig.upper_bd_int64 * tf.ones_like(self.inner_value, dtype='int64')
        x = PrivateTensor(owner=self.owner, inner_value=inner_value, module=self.module, fixedpoint=self.fixedpoint,
                          op_map=self.op_map)
        return x

    def concat(self, other, axis):
        if self.owner == other.owner:
            fixed_point = min(self.fixedpoint, other.fixedpoint)
            x = self.dup_with_precision(fixed_point)
            y = other.dup_with_precision(fixed_point)
            with tf.device(self.owner):
                inner_value = tf.concat([x.inner_value, y.inner_value], axis=axis)
            return PrivateTensor(self.owner, fixedpoint=fixed_point, inner_value=inner_value, module=self.module,
                                 op_map=self.op_map)
        else:
            x = self.to_SharedPair(other.owner)
            y = other.to_SharedPair(self.owner)
            return x.concat(y, axis=axis)

    def stack(self, other, axis):
        if self.owner == other.owner:
            fixed_point = min(self.fixedpoint, other.fixedpoint)
            x = self.dup_with_precision(fixed_point)
            y = other.dup_with_precision(fixed_point)
            with tf.device(self.owner):
                inner_value = tf.stack([x.inner_value, y.inner_value], axis=axis)
            return PrivateTensor(self.owner, fixedpoint=fixed_point, inner_value=inner_value, module=self.module,
                                 op_map=self.op_map)
        else:
            x = self.to_SharedPair(other.owner)
            y = other.to_SharedPair(self.owner)
            return x.stack(y, axis=axis)

    def to_compress_tensor(self, dtype: tf.dtypes.DType = tf.int64) -> tf.Tensor:
        with tf.device(self.owner):
            return self.to_SharedTensor().to_compress_tensor(dtype=dtype)

    def decompress_from(self, y: tf.Tensor, shape):
        with tf.device(self.owner):
            st = self.to_SharedTensor_like()
            st.decompress_from(y, shape=shape)
            self.inner_value = st.inner_value

    def dup_with_precision(self, new_fixedpoint: int):
        y = super(PrivateTensor, self).dup_with_precision(new_fixedpoint)
        return PrivateTensor.from_PrivteTensorBase(y, op_map=self.op_map)

    def reduce_sum(self, axis, keepdims=False):
        with tf.device(self.owner):
            inner_value = tf.reduce_sum(self.inner_value, axis=axis, keepdims=keepdims)
        return PrivateTensor(owner=self.owner, inner_value=inner_value,
                             fixedpoint=self.fixedpoint, module=self.module, op_map=self.op_map)

    def to_private(self, owner):
        owner = get_device(owner)
        if owner == self.owner:
            return self
        else:
            StfEqualWarning("owner", "self.owner", owner, self.owner)
            with tf.device(owner):
                inner_value = tf.identity(self.inner_value)
            return PrivateTensor(owner=owner, inner_value=inner_value,
                                 fixedpoint=self.fixedpoint, module=self.module, op_map=self.op_map)

    def load_from_numpy(self, x: np.ndarray):
        with tf.device(self.owner):
            self.load_from_tf_tensor(x)

    def _convert_result_type_(self, result):
        if isinstance(result, PrivateTensorBase):
            return self.from_PrivteTensorBase(x=result, op_map=self.op_map)
        elif isinstance(result, SharedPairBase):
            return SharedPair.from_SharedPairBase(result)
        else:
            raise StfTypeException("result", "PrivateTensorBase or SharedPairBase", type(result))

    def __add__(self, other):
        result = self.op_map['add'](self, other)
        return self._convert_result_type_(result)

    def __sub__(self, other):
        result = self.op_map['sub'](self, other)
        return self._convert_result_type_(result)

    def __mul__(self, other):
        result = self.op_map['mul'](self, other)
        return self._convert_result_type_(result)

    def __rmul__(self, other):
        result = self.op_map['rmul'](other, self)
        return self._convert_result_type_(result)

    def __truediv__(self, other):
        result = self.op_map['truediv'](self, other)
        return self._convert_result_type_(result)

    def __matmul__(self, other):
        result = self.op_map['matmul'](self, other)
        return self._convert_result_type_(result)

    def __pow__(self, power):
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

    def __invert__(self):
        """
        :return:
        """
        if self.module is not None:
            raise StfCondException("self.module is None", "self.module={}".format(self.module))
        else:
            inner_value = (2 ** (2 * self.fixedpoint)) // self.inner_value
            return PrivateTensor(owner=self.owner, fixedpoint=self.fixedpoint,
                                 inner_value=inner_value, module=self.module)

    def __lt__(self, other):
        result = self.op_map['lt'](self, other)
        return self._convert_result_type_(result)

    def __le__(self, other):
        result = self.op_map['le'](self, other)
        return self._convert_result_type_(result)

    def __gt__(self, other):
        result = self.op_map['gt'](self, other)
        return self._convert_result_type_(result)

    def __ge__(self, other):
        result = self.op_map['ge'](self, other)
        return self._convert_result_type_(result)


class PrivateVariable(PrivateTensor):
    def __init__(self, owner, fixedpoint: int = 14, initial_inner_value=None, module: int = None, op_map={}):
        super(PrivateVariable, self).__init__(owner=owner, fixedpoint=fixedpoint, module=module, op_map=op_map)
        if initial_inner_value is not None:
            with tf.device(self.owner):
                if module is None:
                    self.inner_value = tf.Variable(initial_value=initial_inner_value, trainable=False)
                else:
                    self.inner_value = tf.Variable(initial_value=initial_inner_value % module, trainable=False)

    def __repr__(self) -> str:
        return 'PrivateVariable(owner={}, fixedpoint={}, module={}, shape={})'.format(self.owner.to_string(),
                                                                                      self.fixedpoint, self.module,
                                                                                      self.shape)

    def assign(self, other: PrivateTensorBase) -> tf.Operation:
        if other.owner != self.owner:
            raise StfEqualException("other.owner", "self.owner", other.owner, self.owner)
        if self.fixedpoint != other.fixedpoint:
            raise Exception("the fixedpoint must be same")
        if self.module != other.module:
            raise Exception("the fixedpoint must be same")
        with tf.device(self.owner):
            assign_op = self.inner_value.assign(value=other.inner_value, read_value=False)
        return assign_op

    def load_from_tf_tensor(self, x: tf.Tensor):
        """

        :param x: tf.Tensor
        :return:
        """
        with tf.device(self.owner):
            if self.module is not None:
                self.inner_value = tf.Variable(
                    initial_value=tf.cast(tf.multiply(x, (1 << self.fixedpoint)), 'int64') % self.module,
                    trainable=False)
            else:
                self.inner_value = tf.Variable(initial_value=tf.cast(tf.multiply(x, (1 << self.fixedpoint)), 'int64'),
                                               trainable=False)

    def load_from_numpy(self, x: np.ndarray, const=False):
        if const:
            super(PrivateVariable, self).load_from_tf_tensor(tf.constant(x))
        else:
            with tf.device(self.owner):
                self.load_from_tf_tensor(tf.constant(x))
