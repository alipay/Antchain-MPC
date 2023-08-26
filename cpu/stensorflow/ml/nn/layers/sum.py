#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : Plus
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 11:51
   Description : description what the main function of this file
"""

from functools import reduce
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.exception.exception import StfEqualException, StfCondException


class Sum(Layer):
    def __init__(self, output_dim, fathers=None):
        fathers_output_dim = set(father.output_dim for father in fathers)
        if len(fathers_output_dim) > 1:
            raise StfCondException("len(fathers_output_dim)==1",
                                   "set(fathers_output_dim)={}".format(fathers_output_dim))
        elif fathers_output_dim != {output_dim}:
            raise StfEqualException("fathers_output_dim", "{output_dim}", fathers_output_dim, {output_dim})
        super(Sum, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w, x):
        return reduce(lambda a, b: a + b, x)

    def pull_back(self, w, x, y, ploss_py):
        ploss_pw = []
        ploss_px = dict(zip(self.fathers, [ploss_py] * len(x)))
        return ploss_pw, ploss_px