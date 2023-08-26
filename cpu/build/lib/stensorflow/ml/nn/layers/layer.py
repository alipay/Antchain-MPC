#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : Layer
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 11:51
   Description : description what the main function of this file
"""

from functools import reduce


class Layer:
    def __init__(self, output_dim, fathers=None):
        self.output_dim = output_dim
        self.fathers = fathers
        self.children = []
        self.w = []  # list
        self.x = []  # list
        self.y = None
        self.ploss_pw = []  # list
        self.ploss_px = {}  # map Layer-> ploss_py * py_px
        self.ploss_py = None

    def func(self, w, x):
        # (w, x)->y
        pass

    def pull_back(self, w, x, y, ploss_py):
        # (w, x, y, ploss_py)-> (ploss_pw, ploss_px)
        pass

    def __hash__(self):
        return sum(hash(q) for q in self.w)

    def __eq__(self, other):
        return self.w == other.w

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)

    def forward(self):
        if self.y is not None:
            return
        for father in self.fathers:
            if isinstance(father, Layer):
                # try:
                father.forward()
                # except Exception as e:

            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))
        self.y = self.func(self.w, self.x)

    def backward(self):
        if len(self.ploss_px) > 0:
            return
        for child in self.children:
            if isinstance(child, Layer):
                child.backward()
            else:
                raise Exception("children must be a Layer")

        if self.ploss_py is None:
            ploss_px_of_children = list(map(lambda childre: child.ploss_px[self], self.children))

            self.ploss_py = reduce(lambda x, y: x + y, ploss_px_of_children)

        self.ploss_pw, self.ploss_px = self.pull_back(self.w, self.x, self.y, self.ploss_py)

    def cut_off(self):
        self.y = None
