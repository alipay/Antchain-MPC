#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : ReLU
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-14 17:36
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateTensorBase
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.operator.relu import relu, drelu_binary, relu_pull_back, relu_local, drelu_local
from typing import Union, List, Dict



class ReLU_bak(Layer):
    def __init__(self, output_dim: int, fathers: List[Layer]):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        if fathers[0].output_dim != output_dim:
            raise Exception("must have fathers[0].output_dim == output_dim")
        super(ReLU, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        return relu(x[0])

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) \
            -> (List[SharedPair], Dict[Layer, SharedPair]):
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        # ploss_px = {self.fathers[0]: selectshare(is_positive(x[0]), ploss_py)}
        ploss_px = {self.fathers[0]: relu_pull_back(x[0], ploss_py)}
        return [], ploss_px




class ReLU(Layer):
    def __init__(self, output_dim: int, fathers: List[Layer]):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        if fathers[0].output_dim != output_dim:
            raise Exception("must have fathers[0].output_dim == output_dim")
        super(ReLU, self).__init__(output_dim=output_dim, fathers=fathers)
        self.drelu_b = None

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        self.drelu_b = drelu_binary(x[0])
        return relu(x[0], drelu_b=self.drelu_b)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) \
            -> (List[SharedPair], Dict[Layer, SharedPair]):
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        # ploss_px = {self.fathers[0]: selectshare(is_positive(x[0]), ploss_py)}
        ploss_px = {self.fathers[0]: relu_pull_back(x[0], ploss_py, self.drelu_b)}
        return [], ploss_px


class ReLU_Local(Layer):
    def __init__(self, output_dim: int, fathers: List[Layer], owner):
        if len(fathers) != 1:
            raise Exception("must have len(fathers) == 1 ")
        if fathers[0].output_dim != output_dim:
            raise Exception("must have fathers[0].output_dim == output_dim")
        self.owner = owner
        super(ReLU_Local, self).__init__(output_dim=output_dim, fathers=fathers)

    def func(self, w, x: List[Union[PrivateTensor, SharedPair]]):
        if len(x) != 1:
            raise Exception("must have len(x) == 1")
        if not isinstance(x[0], PrivateTensorBase):
            x = x[0].to_private(owner=self.owner)
        else:
            x = x[0]
        return relu_local(x)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: PrivateTensor,
                  ploss_py: Union[PrivateTensor, SharedPair]) \
            -> (List[PrivateTensor], Dict[Layer, PrivateTensor]):
        if len(x) != 1:
            raise Exception("must have len(x)==1")
        if not isinstance(x[0], PrivateTensorBase):
            x = x[0].to_private(owner=self.owner)
        else:
            x = x[0]
        ploss_py = ploss_py.to_private(owner=self.owner)
        ploss_px = drelu_local(x) * ploss_py
        ploss_px = {self.fathers[0]: ploss_px}
        return [], ploss_px
