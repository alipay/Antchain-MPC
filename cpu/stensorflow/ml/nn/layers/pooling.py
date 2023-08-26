#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group Copyright (c) 2021
   Copyright 2016 The TensorFlow Authors
   All Rights Reserved.
"""
from stensorflow.ml.nn.layers.layer import Layer
from typing import Union, List
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from tensorflow.python.keras.utils import conv_utils
from stensorflow.basic.operator.poolingop import avg_pool2d, sum_pool2d_grad, max_pool2d, max_pool2d_back
import tensorflow as tf
from stensorflow.global_var import StfConfig




PModel = StfConfig.pool_module

class AveragePooling2D(Layer):
    """
    Average pooling operation for spatial data.
     Argument:
        output_dim:
        fathers:
        pool_size:
            An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window.
            Can be a single integer to specify the same value for all spatial dimensions.
        strides:
            An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for all spatial dimensions.
        padding:
            A string. The padding method, only support `"VALID"`
        data_format:
            default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
    """
    def __init__(self, output_dim,
                 fathers: List[Layer],
                 pool_size=(2, 2),
                 strides=None,
                 padding='VALID',
                 data_format=None):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        if data_format is None:
            data_format = "NHWC"
        if strides is None:
            strides = pool_size
        # 初始化参数
        # 池化尺寸
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = padding
        self.data_format = data_format
        # 保存传入数据的shape
        self.input_shape = fathers[0].output_dim
        if not output_dim:
            output_dim = self.compute_shape()
        super(AveragePooling2D, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        """
        Forward propagation
        :param w: None
        :param x: inputs， PrivateTensor or SharedPair
        :return: Has the same type of inputs
        """
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        if self.data_format == 'NHWC':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        # 更新self.input_shape
        self.input_shape = x[0].shape
        return avg_pool2d(input=x[0], ksize=pool_shape, strides=strides, padding=self.padding,
                          data_format=self.data_format)

    def pull_back(self,
                  w: List[SharedPair],
                  x: List[Union[PrivateTensor, SharedPair]],
                  y: SharedPair,
                  ploss_py: SharedPair):
        """
        Back propagation
        :param w: None
        :param x: inputs
        :param y:
        :param ploss_py:
        :return:
        """
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        if self.data_format == 'NHWC':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        # 调用反向传播
        # 计算反向传播系数
        N = self.pool_size[0] * self.pool_size[1]
        # 更新ploss_py
        ploss_py = ploss_py / N
        res = sum_pool2d_grad(input_shape=self.input_shape, out_backprop=ploss_py,
                              ksize=pool_shape, strides=strides, padding=self.padding)
        ploss_px = {self.fathers[0]: res}
        return [], ploss_px

    def compute_shape(self):
        if self.data_format == "NHWC":
            rows = self.input_shape[0]
            cols = self.input_shape[1]
        else:
            rows = self.input_shape[1]
            cols = self.input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding.lower(),
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding.lower(),
                                             self.strides[1])
        if self.data_format == "NHWC":
            return [rows, cols, self.input_shape[2]]
        else:
            return [self.input_shape[0], rows, cols]


class AveragePooling2DLocal(Layer):
    def __init__(self, output_dim,
                 fathers: List[Layer],
                 owner,
                 pool_size=(2, 2),
                 strides=None,
                 padding='VALID',
                 data_format=None):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")

        if data_format is None:
            data_format = "NHWC"
        if strides is None:
            strides = pool_size
        # 初始化参数
        self.owner = owner
        # 池化尺寸
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = padding
        self.data_format = data_format
        # 保存传入数据的shape
        self.input_shape = None
        self.input_shape = fathers[0].output_dim
        if not output_dim:
            output_dim = self.compute_shape()
        super(AveragePooling2DLocal, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w: List[PrivateTensor], x: [PrivateTensor]):
        """
        Forward propagation
        :param w: None
        :param x: inputs
        :return: Has the same type of inputs
        """
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        if self.data_format == 'NHWC':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        x = x[0].to_private(owner=self.owner)  # x为PrivaterTensor
        # 更新self.input_shape
        self.input_shape = x.shape
        # return PrivateTensor
        res = avg_pool2d(input=x, ksize=pool_shape, strides=strides, padding=self.padding,
                         data_format=self.data_format)
        # 下面的代码是需要进行逐层测试时，将需要向下一层传播的信息转化为ShairPair，来符合整体框架使用的
        # ans = SharedPair(ownerL="L", ownerR="R", shape=res.shape)
        # ans.load_from_tf_tensor(res.to_tf_tensor())
        # return ans
        return res

    def pull_back(self,
                  w: List[PrivateTensor],
                  x: [PrivateTensor],
                  y: SharedPair,
                  ploss_py: SharedPair):
        """
        Back propagation
        :param w: None
        :param x: inputs
        :param y:
        :param ploss_py:
        :return:
        """
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        if self.data_format == 'NHWC':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        # 调用反向传播
        # 计算反向传播系数
        N = self.pool_size[0] * self.pool_size[1]
        fix = ploss_py.fixedpoint
        # 将数据转化为tensor
        ploss_py = ploss_py.to_tf_tensor(owner=self.owner)
        ploss_py = ploss_py/N
        # 将数据扩大
        ploss_py = ploss_py*(2**fix)
        ploss_py = tf.cast(ploss_py, dtype=tf.int64)
        # compute
        with tf.device(self.owner):
            res = PModel.sum_pool_grad(self.input_shape, grad=ploss_py,
                                       ksize=pool_shape, strides=strides, padding=self.padding)
        # 将数据缩小
        res = res/(2**fix)
        zx = PrivateTensor(owner=self.owner)
        zx.load_from_tf_tensor(res)
        # 下面的代码是需要进行逐层测试时，将需要向下一层传播的信息转化为ShairPair，来符合整体框架使用的
        # zx = SharedPair(ownerL="L", ownerR="R", shape=res.shape)
        # zx.load_from_tf_tensor(res)
        # 将数据转化为private
        ploss_px = {self.fathers[0]: zx}
        return [], ploss_px

    def compute_shape(self):
        # 假设采用的是更新后的input_shape
        if self.data_format == "NHWC":
            rows = self.input_shape[0]
            cols = self.input_shape[1]
        else:
            rows = self.input_shape[1]
            cols = self.input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding.lower(),
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding.lower(),
                                             self.strides[1])
        if self.data_format == "NHWC":
            return [rows, cols, self.input_shape[2]]
        else:
            return [self.input_shape[0], rows, cols]



class MaxPooling2D(Layer):
    """
    Average pooling operation for spatial data.
     Argument:
        output_dim:
        fathers:
        pool_size:
            An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window.
            Can be a single integer to specify the same value for all spatial dimensions.
        strides:
            An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for all spatial dimensions.
        padding:
            A string. The padding method, only support `"VALID"`
        data_format:
            default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
    """
    def __init__(self, output_dim,
                 fathers: List[Layer],
                 pool_size=(2, 2)):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        self.ksize = [1, pool_size[0], pool_size[1], 1]
        # 保存传入数据的shape
        self.input_shape = fathers[0].output_dim
        if not output_dim:
            output_dim = self.compute_shape()
        super(MaxPooling2D, self).__init__(output_dim=output_dim, fathers=fathers)
        self.pool_size = pool_size
        self.index_list = None

    def func(self, w: List[SharedVariablePair], x: List[SharedPair]):
        """
        Forward propagation
        :param w: None
        :param x: inputs， PrivateTensor or SharedPair
        :return: Has the same type of inputs
        """
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        y, self.index_list = max_pool2d(x[0], ksize=self.ksize, return_s=True)
        return y

    def pull_back(self,
                  w: List[SharedPair],
                  x: List[Union[PrivateTensor, SharedPair]],
                  y: SharedPair,
                  ploss_py: SharedPair):
        """
        Back propagation
        :param w: None
        :param x: inputs
        :param y:
        :param ploss_py:
        :return:
        """
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        res = max_pool2d_back(ploss_py=ploss_py, ksize=self.ksize, index_list=self.index_list)
        ploss_px = {self.fathers[0]: res}
        return [], ploss_px

    def compute_shape(self):
        return [self.input_shape[0]//self.ksize[1], self.input_shape[1]//self.ksize[2], self.input_shape[2]]









