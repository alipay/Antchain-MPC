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

class GraphConv(Layer):
    def __init__(self, output_dim, fathers, adjoint_matrix, with_b=True):
        if fathers is None:
            raise StfNoneException("fathers")
        if fathers == []:
            raise StfCondException("fathers != []", "fathers == []")
        super(GraphConv, self).__init__(output_dim=output_dim, fathers=fathers)
        self.with_b = with_b
        self.adjoint_matrix = adjoint_matrix
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")
            wi = SharedVariablePair(ownerL="L", ownerR="R", shape=[father.output_dim, output_dim])
            wi.load_from_numpy(
                np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))
            self.w += [wi]

        if with_b:
            b = SharedVariablePair(ownerL="L", ownerR="R", shape=[output_dim])
            b.load_from_numpy(np.zeros([output_dim]))
            self.w += [b]



    def __str__(self):
        return "GraphConv Layer of output_dim={}".format(self.output_dim)

    def __repr__(self):
        return self.__str__()

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if len(w) != len(x) + 1:
            raise Exception("must have len(w)==len(x)+1")

        y = x[0] @ w[0]
        y = y.dup_with_precision(x[0].fixedpoint)
        for i in range(1, len(x)):
            y = y + x[i] @ w[i]
        y = self.adjoint_matrix @ y
        if self.with_b:
            y = y + self.w[len(x)]
        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []
        # Z= sum_i X_i W_i   Y = AZ
        ploss_pz = self.adjoint_matrix.transpose() @ ploss_py
        for i in range(len(x)):
            ploss_pxi = ploss_pz @ w[i].transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]

            ploss_pwi = x[i].transpose() @ ploss_pz
            ploss_pwi = ploss_pwi.dup_with_precision(x[0].fixedpoint)
            ploss_pwi = ploss_pwi / batch_size
            ploss_pw += [ploss_pwi.dup_with_precision(x[0].fixedpoint)]

        ploss_px = dict(zip(self.fathers, list_ploss_px))

        if self.with_b:
            ploss_pb = ploss_py.reduce_sum(axis=[0]) / batch_size
            ploss_pw += [ploss_pb.dup_with_precision(x[0].fixedpoint)]

        return ploss_pw, ploss_px
