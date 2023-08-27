#!/usr/bin/env python
# coding=utf-8

from stensorflow.ml.nn.networks.NN import NN
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.relu import ReLU
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.ml.nn.layers.loss import CrossEntropyLossWithSoftmax
from stensorflow.ml.nn.layers.dense import Dense
from stensorflow.random import random



class NETWORKA(NN):
    def __init__(self, feature, label):
        super(NETWORKA, self).__init__()
        # input layer, init data；
        layer = Input(dim=28*28, x=feature)
        self.addLayer(layer)
        # 全连接层
        # a 28*28 × 128 linear layer
        layer = Dense(output_dim=128, fathers=[layer], activate='relu')
        self.addLayer(layer)
        # Relu Layer
        # layer = ReLU(output_dim=128, fathers=[layer])
        # self.addLayer(layer)
        # 全连接层
        # a 128 × 128 linear layer
        layer = Dense(output_dim=128, fathers=[layer], activate='relu')
        self.addLayer(layer)
        # Relu Layer
        # layer = ReLU(output_dim=128, fathers=[layer])
        # self.addLayer(layer)
        # 100x10
        layer = Dense(output_dim=10, fathers=[layer])
        self.addLayer(layer)
        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失计算
        layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
        self.addLayer(ly=layer_loss)

    def predict(self, x, out_prob=True):
        self.cut_off()
        # 输入层
        l_input = self.layers[0]
        assert isinstance(l_input, Input)
        l_input.replace(x)
        self.layers[0] = l_input
        # 输出层
        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        else:
            ly.forward()
        if out_prob:
            return ly.y
        else:
            return ly.score


    def replace_weight(self, keras_weight):
        i = 0
        for ly in self.layers:
            if isinstance(ly, Dense):
                # kernel1 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i].shape)
                # kernel1.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel1
                # kernel2 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i+1].shape)
                # kernel2.load_from_numpy(keras_weight[i+1])
                # ly.w[1] = kernel2
                # 分割
                kernel1 = ly.w[0]
                assert isinstance(kernel1, SharedVariablePair)
                kernel1.load_from_tf_tensor(keras_weight[i])
                kernel2 = ly.w[1]
                assert isinstance(kernel2, SharedVariablePair)
                kernel2.load_from_tf_tensor(keras_weight[i+1])
                i += 2


