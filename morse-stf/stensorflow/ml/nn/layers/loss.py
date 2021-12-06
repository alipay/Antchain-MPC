#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : Loss
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 21:26
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.basic.operator.softmax import softmax
from stensorflow.basic.operator.sigmoid import sigmoid_local, sigmoid_sin as sigmoid
from stensorflow.basic.basic_class.pair import SharedPair

class Loss(Layer):
    BinaryCrossEntropyLossWithSigmoid = 0
    CrossEntropyLossWithSoftmax = 1

    def __init__(self, fathers):
        super(Loss, self).__init__(output_dim=[1], fathers=fathers)


class BinaryCrossEntropyLossWithSigmoid(Loss):
    def __init__(self, layer_score, layer_label):

        fathers = [layer_score, layer_label]
        super(BinaryCrossEntropyLossWithSigmoid, self).__init__(fathers=fathers)
        self.M = 512

    def forward(self):
        for father in self.fathers:
            if isinstance(father, Layer):
                father.forward()
            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))

        self.score = self.x[0]
        self.label = self.x[1]
        self.y = sigmoid(self.score, M=self.M).dup_with_precision(new_fixedpoint=self.score.fixedpoint)

    def backward(self):
        self.ploss_pw = []
        self.ploss_px = {self.fathers[0]: self.y - self.label, self.fathers[1]: -self.score}


class BinaryCrossEntropyLossWithSigmoidLocal(Loss):
    def __init__(self, layer_score: Layer, layer_label: Layer, owner):

        fathers = [layer_score, layer_label]
        super(BinaryCrossEntropyLossWithSigmoidLocal, self).__init__(fathers=fathers)
        self.owner = owner

    def forward(self):

        for father in self.fathers:
            if isinstance(father, Layer):
                father.forward()
            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))

        self.label = self.x[1]
        self.score = self.x[0].to_private(owner=self.owner)

        self.y = sigmoid_local(self.score).dup_with_precision(new_fixedpoint=self.score.fixedpoint)

    def backward(self):
        self.ploss_pw = []
        self.ploss_px = {self.fathers[0]: self.y - self.label, self.fathers[1]: -self.score}


class CrossEntropyLossWithSoftmax(Loss):
    def __init__(self, layer_score, layer_label):

        fathers = [layer_score, layer_label]
        super(CrossEntropyLossWithSoftmax, self).__init__(fathers=fathers)

    def forward(self):
        for father in self.fathers:
            if isinstance(father, Layer):
                father.forward()
            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))
        # dense层的输出值
        self.score = self.x[0]
        # 原始label信息
        self.label = self.x[1]
        self.y = softmax(self.score)

    def backward(self):
        self.ploss_pw = []
        self.ploss_px = {self.fathers[0]: self.y - self.label, self.fathers[1]: -self.score}



class CrossEntropyLossWithSoftmax_bak(Loss):
    def __init__(self, layer_score, layer_label):

        fathers = [layer_score, layer_label]
        super(CrossEntropyLossWithSoftmax, self).__init__(fathers=fathers)

    def forward(self):
        for father in self.fathers:
            if isinstance(father, Layer):
                father.forward()
            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))
        # dense层的输出值
        self.score = self.x[0]
        # 原始label信息
        self.label = self.x[1]
        self.y = softmax(self.score)
        self.ones = self.score.ones_like()


    def backward(self):
        self.ploss_pw = []
        self.ploss_px = {self.fathers[0]: 0.001*(self.ones - 2 * self.label), self.fathers[1]: -self.score}


class CrossEntropyLossWithSoftmaxLocal(Loss):
    def __init__(self, layer_score: Layer, layer_label: Layer, owner):

        fathers = [layer_score, layer_label]
        super(CrossEntropyLossWithSoftmaxLocal, self).__init__(fathers=fathers)
        self.owner = owner

    def forward(self):
        for father in self.fathers:
            if isinstance(father, Layer):
                father.forward()
            else:
                raise Exception("father must be a layer")
        self.x = list(map(lambda father: father.y, self.fathers))

        self.score = self.x[0].to_private(owner=self.owner)
        self.label = self.x[1]
        self.y = softmax(self.score)

    def backward(self):
        self.ploss_pw = []
        self.ploss_px = {self.fathers[0]: self.y - self.label, self.fathers[1]: -self.score}
