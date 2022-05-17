#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : graphconv
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/4/22 下午3:40
   Description : description what the main function of this file
"""
from stensorflow.ml.nn.layers.layer import Layer
import numpy as np
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateVariable
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.basic_class.base import get_device
from typing import Union, List
from stensorflow.basic.protocol.bilinear_triangle import BiliinearTriangle
import pandas as pd
from stensorflow.exception.exception import StfNoneException, StfCondException
import tensorflow as tf

class GraphConv(Layer):
    def __init__(self, fathers, output_dim, adjacency_matrix: PrivateTensor, norm='both',
                 weight=True, bias=True, allow_zero_in_degree=True, train_mask=None):

        if norm not in ("none", "both" or "right"):
            raise StfCondException('norm in Set("none", "both" or "right")', 'norm={}'.format(norm))
        if fathers is None:
            raise StfNoneException("fathers")
        if fathers == []:
            raise StfCondException("fathers != []", "fathers == []")
        if not isinstance(adjacency_matrix, PrivateTensor):
            raise StfCondException("adjacency_matrix is PrivateTensor", "adjacency_matrix is {}".format(adjacency_matrix))
        super(GraphConv, self).__init__(output_dim=output_dim, fathers=fathers)
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.adjacency_matrix = adjacency_matrix
        self.train_mask = train_mask

        if norm == 'both':
            with tf.device(self.adjacency_matrix.owner):
                adjacency_matrix = self.adjacency_matrix.to_tf_tensor()
                out_degrees = tf.reduce_sum(adjacency_matrix, axis=0, keepdims=True)
                in_degrees = tf.reduce_sum(adjacency_matrix, axis=1, keepdims=True)
                out_degrees = tf.clip_by_value(tf.cast(out_degrees, "float64"), clip_value_min=1.0,
                                               clip_value_max=np.inf)
                in_degrees = tf.clip_by_value(tf.cast(in_degrees, 'float64'), clip_value_min=1.0,
                                              clip_value_max=np.inf)
                isqrt_out_degrees = tf.pow(out_degrees, -1 / 2)
                isqrt_in_degrees = tf.pow(in_degrees, -1 / 2)
                isqrt_in_degrees = tf.reshape(isqrt_in_degrees, [1, -1])
                isqrt_out_degrees = tf.reshape(isqrt_out_degrees, [-1, 1])
                normalized_adjacency_matrix = isqrt_in_degrees * adjacency_matrix * isqrt_out_degrees
                self.normalized_adjacency_matrix = PrivateTensor(owner=self.adjacency_matrix.owner)
                self.normalized_adjacency_matrix.load_from_tf_tensor(normalized_adjacency_matrix)
        elif norm == 'right':
            with tf.device(self.adjacency_matrix.owner):
                adjacency_matrix = self.adjacency_matrix.to_tf_tensor()
                in_degrees = tf.reduce_sum(adjacency_matrix, axis=1, keepdims=True)
                in_degrees = tf.clip_by_value(tf.cast(in_degrees, 'float32'), clip_value_min=1.0,
                                              clip_value_max=np.inf)
                inv_in_degrees = 1.0 / in_degrees
                inv_in_degrees = tf.reshape(inv_in_degrees, [1, -1])
                normalized_adjacency_matrix = inv_in_degrees * adjacency_matrix
                self.normalized_adjacency_matrix = PrivateTensor(owner=self.adjacency_matrix.owner)
                self.normalized_adjacency_matrix.load_from_tf_tensor(normalized_adjacency_matrix)
        else:
            self.normalized_adjacency_matrix = self.adjacency_matrix

        self.weight = weight
        self.bias = bias
        if self.weight:
            for father in fathers:
                if not isinstance(father, Layer):
                    raise Exception("father must be a layer")
                wi = SharedVariablePair(ownerL="L", ownerR="R", shape=[father.output_dim, output_dim])
                wi.load_from_numpy(
                    np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))
                self.w += [wi]

        if self.bias:
            b = SharedVariablePair(ownerL="L", ownerR="R", shape=[1, output_dim])
            b.load_from_numpy(np.zeros([1, output_dim]))
            print("b=", b)
            self.w += [b]

    def __str__(self):
        return "GraphConv Layer of output_dim={}".format(self.output_dim)

    def __repr__(self):
        return self.__str__()

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if (len(w) != len(x) + 1) and (len(w) != len(x)):
            # print("w=", w)
            # print("x=", x)
            raise Exception("must have len(w)==len(x)+1 or len(w)==len(x)")

        y = x[0] @ w[0]
        y = y.dup_with_precision(x[0].fixedpoint)
        for i in range(1, len(x)):
            y = y + x[i] @ w[i]
        y = self.normalized_adjacency_matrix.transpose() @ y
        if self.bias:
            y = y + self.w[len(x)]
        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []
        # Z= sum_i X_i W_i   Y = A^T Z+b
        # print("line 112@graphconv.py, self.normalized_adjacency_matrix=",self.normalized_adjacency_matrix)
        # print("ploss_py=", ploss_py)
        ploss_pz = self.normalized_adjacency_matrix @ ploss_py
        for i in range(len(x)):
            ploss_pxi = ploss_pz @ w[i].transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]

            ploss_pwi = x[i].transpose() @ ploss_pz
            ploss_pwi = ploss_pwi.dup_with_precision(x[0].fixedpoint)
            if self.train_mask is None:
                ploss_pwi = ploss_pwi / batch_size
            else:
                ploss_pwi = ploss_pwi / sum(self.train_mask)
            ploss_pw += [ploss_pwi.dup_with_precision(x[0].fixedpoint)]

        ploss_px = dict(zip(self.fathers, list_ploss_px))

        if self.bias:
            if self.train_mask is None:
                ploss_pb = ploss_py.reduce_sum(axis=[0],keepdims=True) / batch_size
            else:
                print("line 137 ploss_py=", ploss_py)
                ploss_pb = ploss_py.reduce_sum(axis=[0], keepdims=True) / sum(self.train_mask)
                print("line 139 ploss_pb=", ploss_pb)
            ploss_pw += [ploss_pb.dup_with_precision(x[0].fixedpoint)]

        return ploss_pw, ploss_px
