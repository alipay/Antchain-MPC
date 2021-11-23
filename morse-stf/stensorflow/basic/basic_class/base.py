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

import tensorflow as tf
import tensorflow.python as python
from stensorflow.global_var import StfConfig
from stensorflow.exception.exception import StfEqualWarning, StfCondException, StfEqualException, StfTypeException, \
    StfDTypeException, StfValueException, StfNoneException, StfException
import numpy as np
from stensorflow.random.random import gen_rint64, gen_rint64_from_seed
from typing import Union
import random
import os
import string

random.seed(0)


def cycle_lshift_tensor(x: tf.Tensor, index: tf.Tensor):
    """
    Cycle left shift x by index.
    :param x: tf.Tensor
    :param index: tf.Tensor where x.shape[:-1]==index.shape
    :return: y: tf.Tensor, where the vector y[i1,..., in-1,:] is the
            cycle left shift of x[i1,..., in-1,:] by index[i1, ...in-1] for any i1, ..., in-1
    """
    if x.shape[:-1] != index.shape:
        raise StfEqualException(var1_name="x.shape[:-1]", var2_name="index.shape",
                                var1=x.shape[:-1], var2=index.shape)
    module = x.shape[-1]
    index = index % module
    one_to_n = tf.constant(list(range(module)), dtype='int64')
    index_ext = (tf.expand_dims(index, axis=-1) + tf.reshape(one_to_n, shape=[1] * len(index.shape) + [module]))
    index_ext = index_ext % one_to_n.shape[0]

    b = tf.gather(params=x, indices=index_ext, axis=len(index.shape),
                  batch_dims=len(index.shape))
    return b


def cycle_rshift_tensor(x: tf.Tensor, index: tf.Tensor):
    """
    Cycle right shift x by index
    :param x: tf.Tensor
    :param index: tf.Tensor where x.shape[:-1]==index.shape
    :return: y: tf.Tensor, where the vector y[i1,..., in-1,:] is the
            cycle right shift of x[i1,..., in-1,:] by index[i1, ...in-1] for any i1, ..., in-1
    """
    return cycle_lshift_tensor(x, -index)


class SharedTensorBase:
    """
    inner_value:  Tensorflow Tensor or Variable of dtype int64 orint32
    module: int
    shape: list
    must have inner_value is not None or shape is not None
    """

    def __init__(self, inner_value=None, module: int = None, shape=None):

        self.module = module
        self.inner_value = inner_value
        if self.inner_value is not None:
            if self.module is not None:
                inner_value %= self.module
            if not isinstance(inner_value, tf.Tensor):
                self.inner_value = tf.constant(inner_value, dtype='int64')
            else:
                self.inner_value = inner_value
            if self.inner_value.dtype != tf.dtypes.int64 and self.inner_value.dtype != tf.dtypes.int32:
                raise StfDTypeException(obj_name="self.inner_value", expect_dtypes=[tf.dtypes.int32, tf.dtypes.int64],
                                        real_dtype=self.inner_value.dtype)

            self.shape = self.inner_value.shape.as_list()
        elif shape is not None:
            self.shape = shape
        else:
            raise StfNoneException(obj_name="inner_value or shape")
        self.name = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    def __repr__(self) -> str:
        return 'SharedTensorBase(module={}, shape={})'.format(self.module, self.shape)

    @property
    def device(self):
        return self.inner_value.device

    def check_inner_value_is_not_None(self):
        if self.inner_value is None:
            raise StfNoneException(obj_name="self.inner_value")

    def check_module_equal(self, other):
        if self.module != other.module:
            raise StfEqualException("self.module", "other.module", self.module, other.module)

    def __getitem__(self, item):
        self.check_inner_value_is_not_None()
        inner_value = self.inner_value[item]
        z = SharedTensorBase(inner_value=inner_value, module=self.module)
        return z

    def random_uniform_adjoint(self, seed=None):
        """

        :param seed: seed for generate random number
        :return:     generate  pseudo-random SharedTensorBase with same shape and
                    module with self
        """
        if seed is not None:
            x = gen_rint64_from_seed(shape=self.shape, seed=seed)
        else:
            x = gen_rint64(self.shape)
            # x = tf.random.uniform(shape=self.shape, minval=-(1 << 63), maxval=(1 << 63) - 1, dtype='int64')

        adjoint = SharedTensorBase(inner_value=x, module=self.module)
        return adjoint

    def random_stateless_uniform_adjoint(self):
        """

        :param seed: seed for generate random number
        :return:     generate  pseudo-random SharedTensorBase with same shape and
                    module with self
        """
        if self.module is not None:
            x = tf.random.stateless_uniform(shape=self.shape, seed=[0, 0], minval=0, maxval=self.module, dtype='int64')
        else:
            x = tf.random.stateless_uniform(shape=self.shape, seed=[0, 0], minval=None, maxval=None, dtype='int64')
        adjoint = SharedTensorBase(inner_value=x, module=self.module)
        return adjoint

    def split(self, size_splits, axis=0, num=None):
        """
        Splits self into a list of sub tensors.

        :param size_splits: Either an integer indicating the number of splits along
          `axis` or a 1-D integer `Tensor` or Python list containing the sizes of
          each output tensor along `axis`. If a scalar, then it must evenly divide
          `value.shape[axis]`; otherwise the sum of sizes along the split axis
          must match that of the `self`.
        :param axis: An integer or scalar `int32` `Tensor`. The dimension along which to
        split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
        :param num: Optional, used to specify the number of outputs when it cannot be
        inferred from the shape of `size_splits`.
        :return:
        """
        self.check_inner_value_is_not_None()
        inner_values = tf.split(self.inner_value, size_splits, axis, num)
        vs = tuple(map(lambda inner_value: SharedTensorBase(inner_value=inner_value, module=self.module), inner_values))
        return vs

    def slice(self, begin, size):
        """
        The usage is refer to tf.slice
        :param begin: An `int32` or `int64` `Tensor`.
        :param size: An `int32` or `int64` `Tensor`.
        :return: A `SharedTensorBase` of the same shape and module as self
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.slice(self.inner_value, begin=begin, size=size)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def to_compress_tensor(self, dtype: tf.dtypes.DType = tf.int64) -> tf.Tensor:
        """
        Compress self to a tf.Tensor of dtype
        :param dtype:  a tf.dtypes.DType object
        :return:  a tf.Tensor of dtype
        """
        self.check_inner_value_is_not_None()
        if self.module is None:
            return tf.bitcast(self.inner_value, dtype)
        else:
            # x = tf.reshape(self.inner_value % self.module, [-1])
            x = tf.reshape(self.inner_value, [-1])
            size = x.shape.as_list()[0]
            capacity = int((8 * dtype.size - 1) // np.log2(self.module))  # how much log(module) in dtype
            new_size = int(np.ceil(1.0 * size / capacity))
            x = tf.pad(x, paddings=[[0, capacity * new_size - size]])
            x = tf.reshape(x, [new_size, capacity])
            y = tf.cast(x, dtype) * tf.constant(np.power(self.module, np.arange(capacity)), dtype=dtype,
                                                shape=[1, capacity])
            return tf.reduce_sum(y, axis=[1])

    def decompress_from(self, y: tf.Tensor, shape=None):
        """
        Decompress from tf.Tensor
        :param y: tf.Tensor
        :param shape:  self.shape
        """
        if shape is not None:
            if shape != self.shape:
                raise Exception(
                    "must have shape is None or shape==self.shape, but shape={}, self.shape={}".format(shape,
                                                                                                       self.shape))
        else:
            shape = self.shape

        if self.module is None:
            self.inner_value = tf.bitcast(y, 'int64')
        else:
            capacity = int((8 * y.dtype.size - 1) // np.log2(self.module))  # how much log(module) in dtype
            div_num = tf.constant(np.power(self.module, np.arange(capacity)), dtype=y.dtype, shape=[1, capacity])
            x = (tf.reshape(y, [-1, 1]) // div_num) % self.module
            x = tf.cast(x, 'int64')
            x = tf.reshape(x, [-1])
            size = int(np.prod(shape))
            x = tf.slice(x, [0], [size])
            self.inner_value = tf.reshape(x, shape)

    def decompress_from_to_new(self, y: tf.Tensor, shape=None):
        """
        Decompress from tf.Tensor
        :param y: tf.Tensor
        :param shape:  self.shape
        """
        if shape is not None:
            if shape != self.shape:
                raise Exception(
                    "must have shape is None or shape==self.shape, but shape={}, self.shape={}".format(shape,
                                                                                                       self.shape))
        else:
            shape = self.shape

        if self.module is None:
            inner_value = tf.bitcast(y, 'int64')
        else:
            capacity = int((8 * y.dtype.size - 1) // np.log2(self.module))  # how much log(module) in dtype
            div_num = tf.constant(np.power(self.module, np.arange(capacity)), dtype=y.dtype, shape=[1, capacity])
            x = (tf.reshape(y, [-1, 1]) // div_num) % self.module
            x = tf.cast(x, 'int64')
            x = tf.reshape(x, [-1])
            size = int(np.prod(shape))
            x = tf.slice(x, [0], [size])
            inner_value = tf.reshape(x, shape)
        return SharedTensorBase(inner_value=inner_value, module=self.module, shape=self.shape)

    def identity(self):
        """
        The usage is same as tf.identity
        :return: A  copy of self
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.identity(self.inner_value)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def squeeze(self, axis=None):
        """
        The usage is same as tf.squeeze
        :param axis: An optional list of `ints`. Defaults to `[]`. If specified, only
                      squeezes the dimensions listed. The dimension index starts at 0. It is an
                      error to squeeze a dimension that is not 1. Must be in the range
                      `[-rank(input), rank(input))`. Must be specified if `input` is a
                      `RaggedTensor`.
        :return:  A `SharedTensorBase`. Has the same type as `self`.
                    Contains the same data as `self`, but has one or more dimensions of
                    size 1 removed.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.squeeze(self.inner_value, axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def rshift(self, index: tf.Tensor):
        """
        Cycle right shift self by index
        :param index: tf.Tensor where x.shape[:-1]==index.shape
        :return: y: SharedTensorBase of same module as self, where the vector y[i1,..., in-1,:] is the
                cycle right shift of x[i1,..., in-1,:] by index[i1, ...in-1] for any i1, ..., in-1
        """
        self.check_inner_value_is_not_None()
        if index.shape != self.shape[0:-1]:
            raise StfEqualException("index.shape", "self.shape[0:-1]", index.shape, self.shape[0:-1])
        index = index % self.shape[-1]
        shifted_reshape = cycle_rshift_tensor(tf.reshape(self.inner_value, shape=[-1, self.shape[-1]]),
                                              tf.reshape(index, shape=[-1]))
        return SharedTensorBase(inner_value=tf.reshape(shifted_reshape, shape=self.shape), module=self.module)

    def lshift(self, index: tf.Tensor):
        """
        Cycle right shift self by i
        :param index: tf.Tensor where x.shape[:-1]==index.shape
        :return: y: SharedTensorBase with same module as self, where the vector y[i1,..., in-1,:] is the
                cycle right shift of x[i1,..., in-1,:] by index[i1, ...in-1] for any i1, ..., in-1
        """
        self.check_inner_value_is_not_None()
        return self.rshift(-index)

    def reshape(self, shape):
        """

        :param shape: list, new shape
        :return:  reshaped SharedTensorBase
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.reshape(self.inner_value, shape)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def transpose(self):
        """

        :return: The transposed SharedTensorBase
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.transpose(self.inner_value)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def reduce_mean(self, axis, keepdims=False):
        """

        :param axis:  The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-len(self.shape),
      len(self.shape))`.
        :param keepdims: If true, retains reduced dimensions with length 1.
        :return: The reduced tensor.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.reduce_mean(self.inner_value, axis=axis, keepdims=keepdims)
        inner_value = tf.cast(inner_value, 'int64')
        if self.module is not None:
            inner_value %= self.module
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def concat(self, other, axis):
        """

        :param other: Another SharedTensorBase
        :param axis: Same as 'axis' in tf.concat
        :return: A `Tensor` resulting from concatenation of the self and other
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.concat(values=[self.inner_value, other.inner_value], axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def stack(self, other, axis):
        """

        :param other:  Another SharedTensorBase
        :param axis: Same as 'axis' in tf.stack
        :return: A stacked SharedTensorBase with the same module of self
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.stack(values=[self.inner_value, other.inner_value], axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def reduce_sum(self, axis, keepdims=False):
        """
        :param axis:  The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-len(self.shape),
      len(self.shape))`.
        :param keepdims: If true, retains reduced dimensions with length 1.
        :return: The reduced tensor.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.reduce_sum(self.inner_value, axis=axis, keepdims=keepdims)
        # inner_value = tf.cast(inner_value, 'int64')
        if self.module is not None:
            inner_value = inner_value % self.module
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def gather(self, indices, axis, batch_dims):
        """

        :param indices:  The index `Tensor`.  Must be one of the following types: `int32`,
      `int64`. Must be in range `[0, params.shape[axis])`.
        :param axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.
        :param batch_dims: An `integer`.  The number of batch dimensions.  Must be less
      than or equal to `rank(indices)`.
        :return: A `SharedTensorBase` with same module as self.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.gather(params=self.inner_value, indices=indices, axis=axis, batch_dims=batch_dims)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def expand_dims(self, axis):
        """
        expand dimension of self, like tf.expand_dims.
        :param axis: Integer specifying the dimension index at which to expand the
      shape of `input`. Given an input of D dimensions, `axis` must be in range
      `[-(D+1), D]` (inclusive).
        :return: A `SharedTensorBase` with same module as self.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.expand_dims(self.inner_value, axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def cumulative_sum(self, axis=-1):
        """
        Compute the cumulative_sum of self along the axis
        :param axis: Integer specifying the dimension index at which to expand the
      shape of `input`. Given an input of D dimensions, `axis` must be in range
      `[-(D+1), D]` (inclusive).
        :return: SharedTensorBase of same shape and module as self
        """
        self.check_inner_value_is_not_None()
        perm = list(range(len(self.shape)))
        perm[0] = perm[axis]
        perm[axis] = 0
        x = tf.transpose(self.inner_value, perm=perm)
        y = tf.scan(lambda a, b: a + b, x, back_prop=False) % self.module
        y = tf.transpose(y, perm=perm)
        return SharedTensorBase(inner_value=y, module=self.module)

    def ones_like(self):
        """
        This operation returns a SharedTensorBase of the
        same module and shape as self with all elements set to 1.
        :return: A SharedTensorBase with all elements set to one.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.ones_like(self.inner_value)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def complement(self):
        """
        For a SharedTensorBas with module 2, return the 1-self.
        :return: A SharedTensorBas with same shape as self and module 2.
        """
        self.check_inner_value_is_not_None()
        if self.module != 2:
            raise StfValueException("self.module", 2, self.module)
        inner_value = tf.math.logical_not(tf.cast(self.inner_value, 'bool'))
        inner_value = tf.cast(inner_value, 'int64')
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def zeros_like(self):
        """
        This operation returns a SharedTensorBase of the
        same module and shape as self with all elements set to 0.
        :return: A SharedTensorBase with all elements set to 0.

        """
        self.check_inner_value_is_not_None()
        inner_value = tf.zeros_like(self.inner_value)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def pad(self, paddings, mode="CONSTANT", constant_values=0):
        """

        :param paddings: A `Tensor` of type `int32`.
        :param mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
        :param constant_values: In "CONSTANT" mode, the scalar pad value to use. Must has type 'int64'.
        :return:  A SharedTensorBase with same module as self.
        """
        self.check_inner_value_is_not_None()
        inner_value = tf.pad(self.inner_value, paddings=paddings, mode=mode, constant_values=constant_values)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def __add__(self, other):
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        inner_value = (self.inner_value + other.inner_value)
        return SharedTensorBase(inner_value=inner_value, module=self.module)

    def __sub__(self, other):
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        inner_value = (self.inner_value - other.inner_value)
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    def __neg__(self):
        self.check_inner_value_is_not_None()
        inner_value = - self.inner_value
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    def __mul__(self, other):
        if not isinstance(other, SharedTensorBase):
            raise StfTypeException("other", SharedTensorBase, type(other))
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        inner_value = (self.inner_value * other.inner_value)
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    def __rmul__(self, other: Union[int, float, np.ndarray, tf.Tensor]):
        self.check_inner_value_is_not_None()
        inner_value = tf.cast(other, 'int64') * self.inner_value
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    def __truediv__(self, other: Union[int, float, np.ndarray, tf.Tensor]):
        self.check_inner_value_is_not_None()
        inner_value = tf.cast(self.inner_value / other, 'int64')
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    __floordiv__ = __truediv__

    def __matmul__(self, other):
        self.check_module_equal(other)
        self.check_inner_value_is_not_None()
        inner_value = tf.matmul(self.inner_value, other.inner_value)
        return SharedTensorBase(inner_value=inner_value,
                                module=self.module)

    def __lshift__(self, other: int):
        self.check_inner_value_is_not_None()
        if other > 62 or other < 0:
            raise StfCondException(cond="0<=other<=62", real="other={}".format(other))
        elif other == 0:
            return self
        else:  # other > 0
            inner_value = (1 << other) * self.inner_value
            return SharedTensorBase(inner_value=inner_value,
                                    module=self.module)

    def __rshift__(self, other: int):
        self.check_inner_value_is_not_None()
        if other < 0:
            raise StfCondException(cond="other>=0", real="other={}".format(other))
        elif other == 0:
            return self
        else:
            inner_value = self.inner_value // (1 << other)
            return SharedTensorBase(inner_value=inner_value,
                                    module=self.module)

    def __pow__(self, power: int):
        self.check_inner_value_is_not_None()
        if not isinstance(power, int):
            raise StfTypeException("power", 'int', type(power))
        elif power < 0:
            raise StfCondException(cond="power>=0", real="power={}".format(power))
        elif power == 0:
            return self.ones_like()
        elif power == 1:
            return self
        elif power == 2:
            inner_value = (self.inner_value ** 2) % self.module
            return SharedTensorBase(inner_value=inner_value, module=self.module)
        else:  # power > 2:
            power_half = power // 2
            y_power_half = self ** power_half
            if power % 2 == 0:
                return y_power_half ** 2
            else:
                return (y_power_half ** 2) * self


    def serialize(self, path=None):
        if path is None:
            raise StfNoneException("path")
        path = os.path.join(path, "serialize")
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, self.name)

        y = tf.io.serialize_tensor(self.inner_value)
        w_op = tf.print(y, output_stream="file://"+path)
        StfConfig.pre_produce_list.append(w_op)
        self.serialize_len=tf.strings.length(tf.io.serialize_tensor(tf.zeros_like(self.inner_value)))

    def unserialize(self, path=None):
        shape = self.shape
        if path is None:
            raise StfNoneException("path")
        path = os.path.join(path, "serialize", self.name)
        # z = tf.compat.v1.data.TextLineDataset(
        #     filenames="file://"+path).make_one_shot_iterator().get_next()
        z = tf.compat.v1.data.FixedLengthRecordDataset(
            filenames="file://"+path, record_bytes=tf.cast(self.serialize_len, 'int64') + 1)\
            .repeat(StfConfig.offline_triple_multiplex).make_one_shot_iterator().get_next()
        z = tf.strings.substr(z, 0, self.serialize_len)
        self.inner_value = tf.reshape(tf.io.parse_tensor(z, 'int64'), shape)




def get_device(owner):
    if isinstance(owner, python.framework.device_spec.DeviceSpecV2):
        device = owner
    elif isinstance(owner, str):
        owner = str.split(owner, ":")  # "L:0"     "L:1"......      "R:0", "R:1"......
        if owner[0] == "L":
            if len(owner) == 1:
                device = StfConfig.workerL[0]
            else:
                device = StfConfig.workerL[int(owner[1])]
        elif owner[0] == "R":
            if len(owner) == 1:
                device = StfConfig.workerR[0]
            else:
                device = StfConfig.workerR[int(owner[1])]
        else:
            raise StfException(msg="owner error: owner={}".format(owner[0]))
    else:
        raise StfTypeException(obj_name="owner", expect_type="str or tf.python.framework.device_spec.DeviceSpecV2",
                               real_type=type(owner))
    return device


class PrivateTensorBase:
    """
    The Class of PrivateTensorBase
    inner_value: A Tensorflow Tensor or Variable of int64
    module: int，
    fixedpoint: int
    owner: String, "L" or "R"
    The represented number = inner_value * 2^{-fixedpoint}
    """

    def __init__(self, owner, fixedpoint: int = None, inner_value=None, module: int = None):
        if fixedpoint is None:
            fixedpoint = StfConfig.default_fixed_point
        if module is not None:
            if not isinstance(module, int):
                raise StfTypeException(obj_name="module", expect_type="int", real_type=type(module))
            else:
                if module <= 0:
                    raise StfCondException(cond="module<=0", real="module={}".format(module))

        self.module = module
        self.owner = get_device(owner)
        self.fixedpoint = fixedpoint
        if inner_value is not None:
            if self.module is not None:
                with tf.device(self.owner):
                    inner_value = inner_value % self.module
            if isinstance(inner_value, (tf.Tensor, tf.Variable)):
                if inner_value.device == self.owner.to_string():
                    self.inner_value = inner_value
                else:
                    with tf.device(self.owner):
                        self.inner_value = tf.identity(inner_value)
            else:
                with tf.device(self.owner):
                    self.inner_value = tf.constant(inner_value)
        self.name = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    @property
    def shape(self):
        return self.inner_value.shape.as_list() if self.inner_value is not None else None

    def __repr__(self) -> str:
        return 'PrivateTensorBase(owner={}, fixedpoint={}, ' \
               'module={}, shape={})'.format(self.owner.to_string(), self.fixedpoint, self.module, self.shape)

    def check_inner_value_is_not_None(self):
        if self.inner_value is None:
            raise StfNoneException(obj_name="self.inner_value")

    def check_module_equal(self, other):
        if self.module != other.module:
            raise StfEqualException("self.module", "other.module", self.module, other.module)

    def check_owner_equal(self, other):
        if self.owner != other.owner:
            raise StfEqualException("self.owner", "other.owner", self.owner, other.owner)


    def slice(self, begin, size):
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_value = tf.slice(self.inner_value, begin, size)
        return PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value, module=self.module)

    def split(self, size_splits, axis=0, num=None):
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_values = tf.split(self.inner_value, size_splits, axis, num)
            pvs = []
            for inner_value in inner_values:
                pvs += [PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint,
                                          inner_value=inner_value, module=self.module)]
        return tuple(pvs)

    def squeeze(self, axis):
        inner_value = tf.squeeze(self.inner_value, axis=axis)
        return PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint,
                                          inner_value=inner_value, module=self.module)

    def load_from_tf_tensor(self, x: Union[tf.Tensor, np.ndarray]):
        """
        Load from tensorflow tensor
        :param x: tf.Tensor
        :return:
        """
        with tf.device(self.owner):
            self.inner_value = tf.cast(tf.multiply(x, (1 << self.fixedpoint)), 'int64')
            if self.module is not None:
                self.inner_value %= self.module
            if self.shape is not None:
                if self.shape != self.inner_value.shape:
                    raise StfEqualException("self.shape", "self.inner_value.shape",
                                            self.shape, self.inner_value.shape)

    def identity(self):
        """
        See tf.identity.
        :return:
        """
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_value = tf.identity(self.inner_value)
        return PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value, module=self.module)

    def load_from_file(self, path: str, record_defaults, batch_size, field_delim=",", skip_row_num=1, skip_col_num=0,
                       repeat=1, clip_value=None, scale=1.0, map_fn=None, output_col_num=None, buffer_size=0):
        """
        Load data from file
        :param path:  absolute path of file in the disk of self.owner
        :param record_defaults: for example [['a'], ['a'], [1.0], [1.0], [1.0]]
        :param batch_size:
        :param field_delim:    field delim between columns
        :param skip_row_num:   skip row number in head of the file
        :param skip_col_num:   skip column number in the file
        :param repeat:         repeat how many times of the file
        :param clip_value:     the features are clip by this value such that |x|<=clip_value
        :param scale:          multiply scale for the  columns
        :param map_fn:         A map function for the columns, for example: lambda x: x[3]*x[4]
        :param output_col_num:   output column number
        :param buffer_size:       buffer size
        :return:
        """
        def clip(r):
            if clip_value is None:
                return r * scale if scale != 1.0 else r
            else:
                return tf.clip_by_value(r * scale, -clip_value, clip_value)

        if output_col_num is None:
            output_col_num = len(record_defaults) - skip_col_num

        with tf.device(self.owner):
            data = tf.compat.v1.data.TextLineDataset(path, buffer_size=buffer_size).skip(skip_row_num)
            data_iter = data.repeat(repeat).batch(batch_size).make_one_shot_iterator()
            data = data_iter.get_next()
            data = tf.reshape(data, [batch_size])
            data = tf.strings.split(data, sep=field_delim).to_tensor(default_value="0.0")
            data = data[:, skip_col_num:]
            data = tf.reshape(data, [batch_size, output_col_num])
            data = tf.strings.to_number(data, out_type='float64')
            data = clip(data)
            if map_fn is not None:
                data = data.map(map_func=map_fn)
        self.load_from_tf_tensor(data)

    def __getitem__(self, item):
        with tf.device(self.owner):
            inner_value = self.inner_value.__getitem__(item)
        return PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value,
                                 module=self.module)

    def load_from_file_withid(self, path: str, record_defaults, batch_size, field_delim=",",
                              skip_row_num=1, id_col_num=0, repeat=1, clip_value=None, use_auto_id_col=False):
        """
        Load data from file, and return the id columns.
        :param path:  absolute path of file in the disk of self.owner
        :param record_defaults: for example [['a'], ['a'], [1.0], [1.0], [1.0]]
        :param batch_size:
        :param field_delim:    field delim between columns
        :param skip_row_num:   skip row number in head of the file
        :param repeat:         repeat how many times of the file
        :param clip_value:     the features are clip by this value such that |x|<=clip_value
        :param id_col_num:          the number of front columns that are id (not feature or label)
        :param use_auto_id_col:  if true, add a auto increase id column [0, 1, 2, ......]
        :return: tf.Tensor of id columns
        """

        def clip(r):
            if clip_value is None:
                return r
            else:
                return tf.clip_by_value(r, -clip_value, clip_value)

        with tf.device(self.owner):
            output_col_num = len(record_defaults) - id_col_num
            data = tf.compat.v1.data.TextLineDataset(path).skip(skip_row_num)
            data_iter = data.repeat(repeat).batch(batch_size).make_one_shot_iterator()
            data = data_iter.get_next()
            data = tf.reshape(data, [batch_size])
            data = tf.strings.split(data, sep=field_delim).to_tensor(default_value="0.0")

            if use_auto_id_col:
                max_value = StfConfig.upper_bd_int64
                idx = tf.compat.v1.data.Dataset.range(max_value).batch(batch_size)
                idx_iter = idx.make_one_shot_iterator()
                idx_batch = tf.reshape(idx_iter.get_next(), [batch_size, id_col_num])
            else:
                idx_batch = data[:, 0:id_col_num]
            data = data[:, id_col_num:]
            data = tf.reshape(data, [batch_size, output_col_num])
            data = tf.strings.to_number(data, out_type='float64')
            data = clip(data)

        self.load_from_tf_tensor(data)
        return idx_batch

    def load_first_line_from_file(self, path: str, col_num, sep=",", dtype='float32'):
        """
        Load the first line of the file
        :param path:  absolute path of file in the disk of self.owner
        :param col_num: columns number of the file
        :param sep:    separation character between columns
        :param dtype:  tf.dtype
        :return:
        """
        with tf.device(self.owner):
            aline = tf.io.read_file(filename=path)
            aline = tf.strings.split(aline, sep="\n")
            aline = aline[0]
            aline = tf.strings.split(aline, sep=sep)
            aline = tf.strings.to_number(aline, dtype)
            aline = tf.reshape(aline, [col_num])
            self.load_from_tf_tensor(aline)

    def to_tf_tensor(self, owner=None, dtype='float64') -> tf.Tensor:
        """
        Transform self to a tf.Tensor
        :param owner:  which machine the tf.Tensor lie
        :param dtype:  tf.Dtype  'float64' or 'int64'
        :return:      tf.Tensor
        """
        self.check_inner_value_is_not_None()
        if owner is None:
            owner = self.owner
        else:
            owner = get_device(owner)
        if owner != self.owner:
            # raise StfEqualException("owner", "self.owner", owner, self.owner)
            StfEqualWarning("owner", "self.owner", owner, self.owner)
        with tf.device(owner):
            if dtype == 'float64':
                return tf.cast(tf.cast(self.inner_value, 'float64') / (2 ** self.fixedpoint), 'float64')
            elif dtype == 'int64':
                return self.inner_value // (2 ** self.fixedpoint)
            else:
                raise StfValueException("dtype", "float64 or int64", dtype)

    def to_tf_str(self, owner=None, precision=StfConfig.to_str_precision, id_col=None):
        """
        Transform self to a tf.Tensor of dtype string.
        :param owner:  which machine the tf.Tensor lie
        :param precision:  An optional `int`. Defaults to `StfConfig.to_str_precision`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
        :param id_col: if not None, add a id column to the data.
        :return: tr.Tensor of dtype string
        """
        self.check_inner_value_is_not_None()
        if owner is None:
            owner = self.owner
        x = self.to_tf_tensor(owner=owner)
        with tf.device(get_device(owner)):
            y = tf.strings.as_string(x, precision=precision)
            if id_col is not None:
                y = tf.concat([id_col, y], axis=1)
            y = tf.compat.v1.reduce_join(y, separator=",", axis=-1)
            return y

    def to_file(self, path: str, separator=",", precision=StfConfig.to_str_precision, dim: int = 1, owner=None,
                dtype='float64', id_col=None):
        """
        Generate a tf.operator, when run it, self is writen to a file in the machine of owner.
        :param path: The file to be written in the machine of self.owner.
        :param separator:  separation character between columns
        :param precision: An optional `int`. Defaults to `StfConfig.to_str_precision`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
        :param dim:   the dimension of self
        :param owner:  'L' or 'R'
        :param dtype:  tf.dtype  'int64' or 'float64'
        :param id_col: if not None, add a id column to the data.
        :return:
        """
        self.check_inner_value_is_not_None()
        if dim > 2 or dim < 1:
            raise StfValueException("dim", "1 or 2", dim)
        if owner is None:
            owner = self.owner
        else:
            owner = get_device(owner)

        with tf.device(owner):
            tf_tensor = self.to_tf_tensor(dtype=dtype)
            if dtype == 'float64':
                tf_tensor = tf.strings.as_string(tf_tensor, precision=precision)
            elif dtype == 'int64':
                tf_tensor = tf.strings.as_string(tf_tensor)
            else:
                raise StfValueException("dtype", 'float64 or int64', dtype)

            if dim == 2 and id_col is not None:
                tf_tensor = tf.concat(values=[id_col, tf_tensor], axis=1)

            weights = tf.compat.v1.reduce_join(tf_tensor, separator=separator, axis=-1)
            if dim == 2:
                weights = tf.compat.v1.reduce_join(weights, separator="\n", axis=-1)
            write_op = tf.io.write_file(filename=path, contents=weights)
            return write_op

    def to_SharedPairBase(self, other_owner):
        """

        :param other_owner:
        :param op_map:
        :return:

        """
        xL = SharedTensorBase(inner_value=self.inner_value, module=self.module)
        xR = -xL.ones_like()  # use one but not zero, for div
        xL += xL.ones_like()
        x = SharedPairBase(ownerL=self.owner, ownerR=other_owner, xL=xL, xR=xR, fixedpoint=self.fixedpoint)
        return x


    def dup_with_precision(self, new_fixedpoint: int):
        """
        Duplicate self to a new fixedpoint.
        :param new_fixedpoint:
        :return:  PrivateTensorBase
        """
        self.check_inner_value_is_not_None()
        if new_fixedpoint == self.fixedpoint:
            return self
        with tf.device(self.owner):
            if new_fixedpoint > self.fixedpoint:
                inner_value = self.inner_value * (1 << (new_fixedpoint - self.fixedpoint))  # left shift
            else:
                inner_value = self.inner_value // (1 << (self.fixedpoint - new_fixedpoint))
            return PrivateTensorBase(self.owner, fixedpoint=new_fixedpoint, inner_value=inner_value,
                                     module=self.module)

    def gather(self, indices, axis, batch_dims):
        """ See tf.gather."""
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_value = tf.gather(params=self.inner_value, indices=indices, axis=axis, batch_dims=batch_dims)
        return PrivateTensorBase(self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value,
                                 module=self.module)

    def stack(self, other, axis):
        """See tf.stack."""
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        self.check_owner_equal(other)

        fixed_point = min(self.fixedpoint, other.fixedpoint)
        x = self.dup_with_precision(fixed_point)
        y = other.dup_with_precision(fixed_point)
        with tf.device(self.owner):
            inner_value = tf.stack([x.inner_value, y.inner_value], axis=axis)
        return PrivateTensorBase(self.owner, fixedpoint=fixed_point, inner_value=inner_value, module=self.module)

    def concat(self, other, axis):
        """See tf.concat"""
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        self.check_owner_equal(other)

        fixed_point = min(self.fixedpoint, other.fixedpoint)
        x = self.dup_with_precision(fixed_point)
        y = other.dup_with_precision(fixed_point)
        with tf.device(self.owner):
            inner_value = tf.concat([x.inner_value, y.inner_value], axis=axis)
        return PrivateTensorBase(self.owner, fixedpoint=fixed_point, inner_value=inner_value, module=self.module)

    def reshape(self, shape):
        """See tf.reshape."""
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_value = tf.reshape(self.inner_value, shape)
            return PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value,
                                     module=self.module)

    def random_uniform_adjoint(self, seed=None):
        """
        generate random uniform adjoint of self.
        :param seed: A seed get from random.get_seed()
        :return:  A uniform random PrivateTensor with same owner, module and fixedpoint as self.
        the value is differential in every time sess.run it
        """
        with tf.device(self.owner):
            x = self.to_SharedTensor_like()
            y = x.random_uniform_adjoint(seed=seed)
            adjoint = PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint, inner_value=y.inner_value,
                                        module=self.module)
            return adjoint

    def zeros_like(self):
        """See tf.zeros_like."""
        with tf.device(self.owner):
            inner_value = tf.zeros_like(self.inner_value)
        return PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value,
                                 module=self.module)

    def ones_like(self, fixedpoint=None):
        """See tf.ones_like."""
        if fixedpoint is None:
            fixedpoint = self.fixedpoint
        with tf.device(self.owner):
            inner_value = (1 << fixedpoint) * tf.ones_like(self.inner_value)
        return PrivateTensorBase(owner=self.owner, fixedpoint=fixedpoint, inner_value=inner_value,
                                 module=self.module)

    def to_SharedTensor(self):
        """
        :return a SharedTensorBase with same inner_value and module with self.
        """
        return SharedTensorBase(inner_value=self.inner_value, module=self.module)

    def to_SharedTensor_like(self):
        """
        :return a SharedTensorBase with inner_value=None, and with same module and shape with self.
        """
        return SharedTensorBase(module=self.module, shape=self.shape)

    def serialize(self, path=None):
        if path is None:
            path = StfConfig.stf_home_workerL if self.owner == StfConfig.workerL[0] else StfConfig.stf_home_workerR
        # print("line 1015 base.py, path=", path)
        path = os.path.join(path, "serialize")
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, self.name)
        # print("line 1015 base.py, path=", path)
        with tf.device(self.owner):
            y = tf.io.serialize_tensor(self.inner_value)
            self.serialize_len = tf.strings.length(tf.io.serialize_tensor(tf.zeros_like(self.inner_value)))
            w_op = tf.print(y, output_stream="file://"+path)
        StfConfig.pre_produce_list.append(w_op)
        # print("StfConfig.pre_produce_list=", StfConfig.pre_produce_list)



    def unserialize(self, path=None):
        shape =self.shape
        if path is None:
            path = StfConfig.stf_home_workerL if self.owner == StfConfig.workerL[0] else StfConfig.stf_home_workerR
        path = os.path.join(path, "serialize", self.name)
        with tf.device(self.owner):
            # z = tf.compat.v1.data.TextLineDataset(
            #     filenames="file://"+path).make_one_shot_iterator().get_next()
            z = tf.compat.v1.data.FixedLengthRecordDataset(
                filenames="file://" + path,
                record_bytes=tf.cast(self.serialize_len, 'int64') + 1).repeat(StfConfig.offline_triple_multiplex).make_one_shot_iterator().get_next()
            z = tf.strings.substr(z, 0, self.serialize_len)
            self.inner_value = tf.reshape(tf.io.parse_tensor(z, 'int64'), shape)
        self.inner_value = tf.reshape(tf.io.parse_tensor(z, 'int64'), shape)


    def __add__(self, other):
        self.check_inner_value_is_not_None()
        self.check_module_equal(other)
        self.check_owner_equal(other)
        with tf.device(self.owner):
            fixed_point = min(self.fixedpoint, other.fixedpoint)
            altered_self = self.dup_with_precision(new_fixedpoint=fixed_point)
            altered_other = other.dup_with_precision(new_fixedpoint=fixed_point)
            inner_value = (altered_self.inner_value + altered_other.inner_value)
            if self.module is not None:
                inner_value %= self.module
            return PrivateTensorBase(owner=self.owner, fixedpoint=fixed_point, inner_value=inner_value,
                                     module=self.module)

    def __neg__(self):
        self.check_inner_value_is_not_None()
        with tf.device(self.owner):
            inner_value = - self.inner_value
            if self.module is not None:
                inner_value = inner_value % self.module
            return PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint, inner_value=inner_value,
                                     module=self.module)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other, fixed_point=None):

        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = (self.inner_value * other.inner_value)
            if self.module is not None:
                inner_value = inner_value % self.module
            w = PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint + other.fixedpoint,
                                  inner_value=inner_value,
                                  module=self.module)
        if not fixed_point:
            fixed_point = max(self.fixedpoint, other.fixedpoint)
            # fixed_point = self.fixedpoint + other.fixedpoint
        return w.dup_with_precision(fixed_point)

    def __matmul__(self, other, fixed_point=None):
        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = tf.matmul(self.inner_value, other.inner_value)
            if self.module is not None:
                inner_value = inner_value % self.module
            w = PrivateTensorBase(owner=self.owner, fixedpoint=self.fixedpoint + other.fixedpoint,
                                  inner_value=inner_value,
                                  module=self.module)
        if not fixed_point:
            fixed_point = max(self.fixedpoint, other.fixedpoint)
            # fixed_point = self.fixedpoint + other.fixedpoint
        return w.dup_with_precision(fixed_point)

    def __lt__(self, other):
        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = tf.cast(self.to_tf_tensor() < other.to_tf_tensor(), 'int64')
        return PrivateTensorBase(owner=self.owner, fixedpoint=0, inner_value=inner_value, module=2)

    def __le__(self, other):
        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = tf.cast(self.to_tf_tensor() <= other.to_tf_tensor(), 'int64')
        return PrivateTensorBase(owner=self.owner, fixedpoint=0, inner_value=inner_value, module=2)

    def __gt__(self, other):
        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = tf.cast(self.to_tf_tensor() > other.to_tf_tensor(), 'int64')
        return PrivateTensorBase(owner=self.owner, fixedpoint=0, inner_value=inner_value, module=2)

    def __ge__(self, other):
        if self.owner != other.owner:
            raise Exception("owner must be same")
        if self.module != other.module:
            raise Exception("module must be same")
        with tf.device(self.owner):
            inner_value = tf.cast(self.to_tf_tensor() >= other.to_tf_tensor(), 'int64')
        return PrivateTensorBase(owner=self.owner, fixedpoint=0, inner_value=inner_value, module=2)


class SharedPairBase:
    """
    A SharedPairBase x is represented by x= (xL+xR mod n)*2^{-fixedpoint},
    where xL is in ownerL and xR is in ownerR, n=xL.module=xR.module.
    :param ownerL: str (for example, "L") or object of python.framework.device_spec.DeviceSpecV2
    :param ownerR: str (for example, "R") or object of python.framework.device_spec.DeviceSpecV2
    :param xL: a SharedTensorBase.
    :param xR: a SharedTensorBase.
    :param fixedpoint: int. default is StfConfig.default_fixed_point
    :param shape:  list of int
    Shape or (xL,xR) must not be None.
    """
    def __init__(self, ownerL, ownerR, xL: SharedTensorBase = None, xR: SharedTensorBase = None,
                 fixedpoint: int = StfConfig.default_fixed_point,
                 shape=None):
        ownerL = get_device(ownerL)
        ownerR = get_device(ownerR)

        if ownerL == ownerR:
            raise StfException(
                "The ownerL and ownerR should be different. But ownerL={}, ownerR={}".format(ownerL, ownerR))
        if shape is not None and xL is not None:
            if shape != xL.shape:
                raise StfEqualException("shape", "xL.shape", shape, xL.shape)
        self.ownerL = ownerL
        self.ownerR = ownerR
        self.fixedpoint = fixedpoint
        if xL is not None:
            if xL.device == self.ownerL.to_string():
                self.xL = xL
            else:
                with tf.device(self.ownerL):
                    self.xL = xL.identity()
        else:
            self.xL = None
        if xR is not None:
            if xR.device == self.ownerR.to_string():
                self.xR = xR
            else:
                with tf.device(self.ownerR):
                    self.xR = xR.identity()
        else:
            self.xR = None
        if xL is not None and xR is not None:
            if xL.module != xR.module:
                raise StfEqualException("xL.module", "xR.module", xL.module, xR.module)
        if shape is not None:
            self.shape = shape
        elif self.xL is not None:
            self.shape = self.xL.shape
        else:
            raise StfNoneException("shape or xL")

    def mirror(self):
        """
        Get the mirrored SharedPairBase of self.
        :return:
        """
        ownerR = self.ownerL
        ownerL = self.ownerR
        xR = self.xL
        xL = self.xR
        return SharedPairBase(ownerL, ownerR, xL, xR, self.fixedpoint)

    def __repr__(self) -> str:
        return 'SharedPairBase(ownerL={},ownerR={}, fixedpoint={}, module={}, shape={})'.format(self.ownerL.to_string(),
                                                                                                self.ownerR.to_string(),
                                                                                                self.fixedpoint,
                                                                                                self.xL.module,
                                                                                                self.shape)

    def __getitem__(self, index):
        xL = self.xL[index]
        xR = self.xR[index]
        z = SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)
        return z

    def split(self, size_splits, axis: int = 0, num=None):
        """
        See tf.split.
        :param size_splits:
        :param axis:
        :param num:
        :return:
        """
        with tf.device(self.ownerL):
            xLs = self.xL.split(size_splits=size_splits, axis=axis, num=num)
        with tf.device(self.ownerR):
            xRs = self.xR.split(size_splits=size_splits, axis=axis, num=num)
        sps = []
        for xL_xR in zip(xLs, xRs):
            sps += [SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR,
                                   xL=xL_xR[0], xR=xL_xR[1], fixedpoint=self.fixedpoint)]
        return tuple(sps)

    def to_SharedTensor_like(self):
        """
        :return:  A SharedTensorBase with inner_value=None and same module and shape with self.
        """
        return SharedTensorBase(module=self.xR.module, shape=self.shape)

    def identity(self, ownerL=None, ownerR=None):
        """
        See tf.identity.
        :param ownerL:
        :param ownerR:
        :return:
        """
        if ownerL is None:
            ownerL = self.ownerL
        if ownerR is None:
            ownerR = self.ownerR
        with tf.device(ownerL):
            xL = self.xL.identity()
        with tf.device(ownerR):
            xR = self.xR.identity()
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def to_private(self, owner) -> PrivateTensorBase:
        """
        transform to a PrivateTensorBase.
        :param owner:
        :return:
        """
        with tf.device(get_device(owner)):
            x = self.xL.inner_value + self.xR.inner_value
            return PrivateTensorBase(owner=owner, fixedpoint=self.fixedpoint, inner_value=x, module=self.xL.module)

    def to_tf_tensor(self, owner) -> tf.Tensor:
        """
        transform self to a tensorflow Tensor.
        :param owner:
        :return:
        """
        with tf.device(get_device(owner)):
            x = self.to_private(owner)
            return x.to_tf_tensor()

    def ones_like(self):
        """
        See tf.ones_like
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.zeros_like()
        with tf.device(self.ownerR):
            xR = self.xR.ones_like()  # << self.fixedpoint
        return SharedPairBase(self.ownerL, self.ownerR, xL, xR, fixedpoint=0)

    def complement(self):
        """
        compute the two's complement of self.
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.complement()
        return SharedPairBase(self.ownerL, self.ownerR, xL, self.xR, fixedpoint=self.fixedpoint)

    def zeros_like(self):
        """
        See tf.zeros_like.
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.zeros_like()
        with tf.device(self.ownerR):
            xR = self.xR.zeros_like()
        return SharedPairBase(self.ownerL, self.ownerR, xL, xR, fixedpoint=self.fixedpoint)

    def cumulative_sum(self, axis=-1):
        """
        Compute the cumulative_sum of self alone the given axis.
        :param axis:
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.cumulative_sum(axis)
        with tf.device(self.ownerR):
            xR = self.xR.cumulative_sum(axis)
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def load_from_tf_tensor(self, x):
        """
        Load data from a tensorflow Tensor.
        :param x:
        :return:
        """
        with tf.device(self.ownerL):
            px = PrivateTensorBase(owner=self.ownerL, fixedpoint=self.fixedpoint)
            px.load_from_tf_tensor(x)
            u = px.to_SharedTensor()
            self.xL = -u.ones_like()
        with tf.device(self.ownerR):
            self.xR = u + u.ones_like()
        self.fixedpoint = px.fixedpoint

    def _align_owner(self, other):
        """
        Align the owner of other with self.
        :param other: A SharedPairBase.
        :return: A SharedPairBase with same value as `other` satisfying ownerL=self.ownerL
        and ownerR=self.ownerR.
        """
        if self.ownerL == other.ownerL and self.ownerR == other.ownerR:
            return other
        elif self.ownerL == other.ownerR and self.ownerR == other.ownerL:
            return other.mirror()
        else:
            raise Exception("ownerL must be same, ownerR must be same.")

    def concat(self, other, axis):
        """
        See tf.concat.
        :param other:
        :param axis:
        :return:
        """
        other = self._align_owner(other)
        fixed_point = min(self.fixedpoint, other.fixedpoint)
        alter_self = self.dup_with_precision(fixed_point)
        other = other.dup_with_precision(fixed_point)
        with tf.device(self.ownerL):
            xL = alter_self.xL.concat(other.xL, axis=axis)
        with tf.device(self.ownerR):
            xR = alter_self.xR.concat(other.xR, axis=axis)
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=fixed_point)

    def stack(self, other, axis):
        """
        See tf.stack.
        :param other:
        :param axis:
        :return:
        """
        other = self._align_owner(other)
        fixed_point = min(self.fixedpoint, other.fixedpoint)
        alter_self = self.dup_with_precision(fixed_point)
        other = other.dup_with_precision(fixed_point)
        with tf.device(self.ownerL):
            xL = alter_self.xL.stack(other.xL, axis=axis)
        with tf.device(self.ownerR):
            xR = alter_self.xR.stack(other.xR, axis=axis)
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=fixed_point)

    def dup_with_precision(self, new_fixedpoint: int):
        """
        Duplicate self to a new SharedPairBase with new fixedpoint.
        :param new_fixedpoint:
        :return:
        """
        if new_fixedpoint > self.fixedpoint:
            with tf.device(self.ownerL):
                xL = self.xL << (new_fixedpoint - self.fixedpoint)  # left shift
            with tf.device(self.ownerR):
                xR = self.xR << (new_fixedpoint - self.fixedpoint)  # left shift
            return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=new_fixedpoint)
        elif new_fixedpoint < self.fixedpoint:
            with tf.device(self.ownerL):
                xL = self.xL >> (self.fixedpoint - new_fixedpoint)  # right shift
            with tf.device(self.ownerR):
                xR = -((-self.xR) >> (self.fixedpoint - new_fixedpoint))  # right shift
            return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=new_fixedpoint)
        else:
            return self

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
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def squeeze(self, axis=None):
        """
        See tf.squeeze.
        :param axis:
        :return:
        """
        with tf.device(self.ownerL):
            xL = self.xL.squeeze(axis=axis)
        with tf.device(self.ownerR):
            xR = self.xR.squeeze(axis=axis)
        return SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR, fixedpoint=self.fixedpoint)

    def to_tf_str(self, owner, precision=StfConfig.to_str_precision, id_col=None):
        """
        Transform self to tf.string.
        :param owner: The owner of tf
        :param precision:
        :param id_col:
        :return:
        """
        return self.to_private(owner=owner).to_tf_str(precision=precision, id_col=id_col)

    def serialize(self):
        with tf.device(StfConfig.workerL[0]):
            self.xL.serialize(StfConfig.stf_home_workerL)
        with tf.device(StfConfig.workerR[0]):
            self.xR.serialize(StfConfig.stf_home_workerR)

    def unserialize(self):
        with tf.device(StfConfig.workerL[0]):
            self.xL.unserialize(StfConfig.stf_home_workerL)
        with tf.device(StfConfig.workerR[0]):
            self.xR.unserialize(StfConfig.stf_home_workerR)

    def random_uniform_adjoint(self, seed=None):
        with tf.device(self.ownerL):
            xL = self.xL.random_uniform_adjoint(seed)
        with tf.device(self.ownerR):
            xR = self.xR.random_uniform_adjoint(seed)
        z = SharedPairBase(ownerL=self.ownerL, ownerR=self.ownerR, xL=xL, xR=xR,
                           fixedpoint=self.fixedpoint)
        return z


    def __add__(self, other):
        other = self._align_owner(other)
        fixed_point = min(self.fixedpoint, other.fixedpoint)
        altered_self = self.dup_with_precision(new_fixedpoint=fixed_point)
        altered_other = other.dup_with_precision(new_fixedpoint=fixed_point)
        with tf.device(self.ownerL):
            xL = altered_self.xL + altered_other.xL
        with tf.device(self.ownerR):
            xR = altered_self.xR + altered_other.xR

        return SharedPairBase(self.ownerL, self.ownerR, xL, xR, fixed_point)

    def __neg__(self):
        return SharedPairBase(self.ownerL, self.ownerR, -self.xL, -self.xR, self.fixedpoint)

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        with tf.device(self.ownerL):
            xL = other * self.xL
        with tf.device(self.ownerR):
            xR = other * self.xR
        return SharedPairBase(self.ownerL, self.ownerR, xL, xR, self.fixedpoint)
