import tensorflow as tf
from stensorflow.basic.basic_class.base import SharedTensorBase
import numpy as np
from typing import Union
from stensorflow.exception.exception import StfTypeException, StfDTypeException


class SharedTensor(SharedTensorBase):
    def __repr__(self) -> str:
        return 'SharedTensor(module={}, shape={})'.format(self.module, self.inner_value.shape)


class SharedVariable(SharedTensor):
    def __init__(self, initial_inner_value=None, module: int = None, shape=None):

        if initial_inner_value is not None:
            super(SharedVariable, self).__init__(module=module, shape=initial_inner_value.shape)
            if module is None:
                self.inner_value = tf.Variable(initial_value=initial_inner_value)
            else:
                self.inner_value = tf.Variable(initial_value=initial_inner_value % module)
        elif shape is not None:
            super(SharedVariable, self).__init__(module=module, shape=shape)

    def __repr__(self) -> str:
        return 'SharedVariable(module={}, shape={})'.format(self.module, self.inner_value.shape)

    def assign(self, other):
        self.check_module_equal(other)
        assign_op = self.inner_value.assign(value=0 * self.inner_value + other.inner_value, read_value=False)
        return assign_op


def sin2pi(x: SharedTensor, fixedpoint, k: Union[int, tf.Tensor] = None):
    """
    sin 2 k pi (x/ (2**fixedpoint))
    :param x:
    :param fixedpoint:
    :param k:
    :return:
    """
    if k is None:
        k = 1
    if isinstance(k, int):
        pass
    elif isinstance(k, tf.Tensor):
        if k.dtype in [tf.dtypes.int8, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64]:
            k = tf.cast(k, 'int64')
        else:
            raise StfDTypeException("k", "tf.dtypes.int8, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64",
                                    k.dtype)
    else:
        raise StfTypeException("k", "int or tf.Tensor", type(k))
    n = (1 << fixedpoint)
    return tf.sin(
        2 * np.pi * tf.cast(k * (x.inner_value % n) % n, 'float64') / n)


def cos2pi(x: SharedTensor, fixedpoint, k: Union[int, tf.Tensor] = None):
    """
    cos 2 k pi x/ (2**fixedpoint)
    :param x:
    :param fixedpoint:
    :param k:
    :return:
    """
    if k is None:
        k = 1
    if isinstance(k, int):
        pass
    elif isinstance(k, tf.Tensor):
        if k.dtype in [tf.dtypes.int8, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64]:
            k = tf.cast(k, 'int64')
        else:
            raise StfDTypeException("k", "tf.dtypes.int8, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64",
                                    k.dtype)
    else:
        raise StfTypeException("k", "int or tf.Tensor", type(k))
    n = (1 << fixedpoint)
    return tf.cos(
        2 * np.pi * tf.cast(k * (x.inner_value % n) % n, 'float64') / n)

