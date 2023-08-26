#!/usr/bin/env python
# coding=utf-8
import numpy as np
from stensorflow.ml.nn.layers.conv2d import *
from stensorflow.ml.nn.layers.pooling import *


class Flatten(Layer):
    """
    Flattens the input. Does not affect the batch size.
    If inputs are shaped `(batch,)` without a channel dimension,
    then flattening adds an extra channel dimension and output shapes are `(batch, 1)`.
    """
    def __init__(self, output_dim, fathers: List[Layer]):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        self.input_shape = fathers[0].output_dim
        # if isinstance(fathers[0], Conv2d) or isinstance(fathers[0], AveragePooling2D):
        if output_dim is None:
            output_dim = np.prod(self.input_shape[:], dtype=int)
        super(Flatten, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        """
        :param w:
        :param x: input
        :return: Has the same type as `x`.
        """
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        # only support data_format is "NHWC"
        # 更新input_shape
        input_shape = x[0].shape  # return a list
        self.input_shape = input_shape  # save shape information
        flattened_dim = np.prod(input_shape[1:], dtype=int)
        # 这里进行了一次隐蔽的类型转换
        y = x[0].reshape([-1, flattened_dim])
        if isinstance(y, SharedPairBase):
            z = SharedPair(ownerL=y.ownerL, ownerR=y.ownerR,
                           xL=y.xL, xR=y.xR, fixedpoint=y.fixedpoint)
        if isinstance(y, PrivateTensorBase):
            z = PrivateTensor.from_PrivteTensorBase(y)
        return z

    def pull_back(self, w, x, y, ploss_py):
        """
        :param w:
        :param x:
        :param y:
        :param ploss_py: loss gradient Delta
        :return:
        """
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        if not self.input_shape:
            raise Exception("input_shape != None")
        ploss_px = ploss_py.reshape(self.input_shape)
        if isinstance(ploss_px, SharedPairBase):
            z = SharedPair(ownerL=ploss_px.ownerL, ownerR=ploss_px.ownerR,
                           xL=ploss_px.xL, xR=ploss_px.xR, fixedpoint=ploss_px.fixedpoint)
        if isinstance(ploss_px, PrivateTensorBase):
            z = PrivateTensor.from_PrivteTensorBase(ploss_px)
        ploss_px = {self.fathers[0]: z}
        return [], ploss_px


class FlattenLocal(Layer):
    def __init__(self, output_dim, fathers: List[Layer], owner):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        self.owner = owner
        self.input_shape = fathers[0].output_dim
        if isinstance(fathers[0], Conv2dLocal) or isinstance(fathers[0], AveragePooling2DLocal):
            output_dim = np.prod(self.input_shape[:], dtype=int)
        super(FlattenLocal, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w: List[SharedVariablePair], x: [PrivateTensor, SharedPair]):
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        x = x[0].to_private(owner=self.owner)
        x = PrivateTensor.from_PrivteTensorBase(x)
        # only support data_format is "NHWC"
        input_shape = x.shape  # return a list
        self.input_shape = input_shape  # save shape information
        flattened_dim = np.prod(input_shape[1:], dtype=int)
        # 这里进行了一次隐蔽的类型转换
        y = x.reshape([-1, flattened_dim])
        y = PrivateTensor.from_PrivteTensorBase(y)
        return y

    def pull_back(self, w, x, y, ploss_py):
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        if not self.input_shape:
            raise Exception("input_shape != None")
        ploss_py = ploss_py.to_private(owner=self.owner)
        ploss_py = PrivateTensor.from_PrivteTensorBase(ploss_py)
        # 一次隐式的类型转换
        ploss_px = ploss_py.reshape(self.input_shape)
        ploss_px = PrivateTensor.from_PrivteTensorBase(ploss_px)
        # 测试单层而坐的转换
        ploss_px = {self.fathers[0]: ploss_px}
        return [], ploss_px
