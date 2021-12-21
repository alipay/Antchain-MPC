#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : Input
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 19:43
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.layers.layer import Layer
# import numpy as np
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from typing import Union, List


# from stensorflow.global_var import workerL, workerR

class Input(Layer):
    """
    Input Layer
    """
    def __init__(self, dim: Union[int, List[int]], x: PrivateTensor, owner=None):
        if isinstance(dim, int):
            if x.shape[1] != dim:
                raise Exception("must have x.shape[1] == dim")
        else:
            if x.shape[1:] != dim:
                raise Exception("must have x.shape[1:] == dim, but x.shape[1:]={}".format(x.shape[1:]))
        super(Input, self).__init__(output_dim=dim, fathers=[])
        self.y = x
        # self.ploss_px = {None: x.zeros_like().transpose()}
        if owner is None:
            self.owner = x.owner
        else:
            self.owner = owner

    def forward(self):
        return

    def backward(self):
        # if len(self.ploss_px) > 0:
        #     return
        for child in self.children:
            if isinstance(child, Layer):
                child.backward()
            else:
                raise Exception("children must be a Layer")
        # self.ploss_px = {None: x.zeros_like().transpose()}

    def replace(self, x: Union[SharedPair, PrivateTensor]):
        if x.shape[1] != self.output_dim:
            raise Exception("must have x.shape[1] == self.output_dim")
        self.y = x
