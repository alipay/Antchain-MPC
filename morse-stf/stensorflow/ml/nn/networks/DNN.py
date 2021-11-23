#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : DNN
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-14 11:52
   Description : description what the main function of this file
"""
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.ml.nn.networks.NN import NN
from typing import List
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.dense import Dense
from stensorflow.ml.nn.layers.loss import Loss, BinaryCrossEntropyLossWithSigmoid, CrossEntropyLossWithSoftmax
from stensorflow.ml.nn.layers.relu import ReLU
from stensorflow.random.random import random_init


class DNN(NN):
    def __init__(self, feature: PrivateTensor, label: PrivateTensor, dense_dims: List[int],
                 feature_another: PrivateTensor = None, loss=None):
        """
        Full-connected Neural Network. Must have feature.owner != label.owner. If
        feature_another is not None, feature_another.owner==label.owner.
        :param feature:            feature in different party of label.owner
        :param label:
        :param dense_dims:  [d0, d1, ..., dn]  where d0=dim(feature) dn=dim(label), di is the
        dimension of i-th hidden layer
        :param feature_another:    feature in same party of label.owner
        :param loss
        """

        if len(dense_dims) < 2:
            raise Exception("must have len(dense_dims)>=2")
        if feature_another is None:
            self.double_feature = False
            if feature.shape[1] != dense_dims[0]:
                raise Exception("must have x.shape[1] == dense_dims[0]")
            super(DNN, self).__init__()
            layer = Input(dim=dense_dims[0], x=feature)
            self.addLayer(ly=layer)
            input_layers = [layer]

        else:
            self.double_feature = True
            if feature.shape[1] + feature_another.shape[1] != dense_dims[0]:
                raise Exception("must have x.shape[1] == dense_dims[0]")
            if feature.owner == feature_another.owner:
                raise Exception("must have feature.owner != feature_another.owner")
            super(DNN, self).__init__()
            input_layer = Input(dim=feature.shape[1], x=feature)
            self.addLayer(ly=input_layer)
            input_layer_another = Input(dim=feature_another.shape[1], x=feature_another)
            self.addLayer(input_layer_another)
            input_layers = [input_layer, input_layer_another]

        for i in range(1, len(dense_dims)):
            if i == 1:
                layer = Dense(output_dim=dense_dims[i], fathers=input_layers)
            else:
                layer = Dense(output_dim=dense_dims[i], fathers=[layer])
            self.addLayer(ly=layer)
            if i < len(dense_dims) - 1:
                layer = ReLU(output_dim=dense_dims[i], fathers=[layer])
                self.addLayer(ly=layer)

        layer_label = Input(dim=dense_dims[-1], x=label)
        self.addLayer(ly=layer_label)

        if loss is None or loss == "BinaryCrossEntropyLossWithSigmoid" or \
                loss == Loss.BinaryCrossEntropyLossWithSigmoid:
            layer_loss = BinaryCrossEntropyLossWithSigmoid(layer_score=layer, layer_label=layer_label)
            self.addLayer(ly=layer_loss)
        elif loss == "CrossEntropyLossWithSoftmax" or loss == Loss.CrossEntropyLossWithSoftmax:
            layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
            self.addLayer(ly=layer_loss)
        else:
            raise Exception("unsupposed loss")

    def predict(self, x, x_another=None, out_prob=True) -> SharedPair:
        self.cut_off()
        l_input = self.layers[0]
        assert isinstance(l_input, Input)
        l_input.replace(x)
        self.layers[0] = l_input

        if self.double_feature:
            l_input_another = self.layers[1]
            if not isinstance(l_input_another, Input):
                raise Exception("l_input_another mast be a Input layer")
            l_input_another.replace(x_another)
            self.layers[1] = l_input_another

        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        else:
            ly.forward()

        if out_prob:
            return ly.y
        else:
            return ly.score

    def predict_to_file(self, sess, x, predict_file_name,
                        batch_num, idx, model_file_machine="R", record_num_ceil_mod_batch_size=0,
                        x_another=None, with_sigmoid=True):
        y_pred = self.predict(x=x, x_another=x_another, out_prob=with_sigmoid)

        id_y_pred = y_pred.to_tf_str(owner=model_file_machine, id_col=idx)
        random_init(sess)

        with open(predict_file_name, "w") as f:
            for batch in range(batch_num - 1):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                f.write(records + "\n")

            records = sess.run(id_y_pred)[0:record_num_ceil_mod_batch_size]
            records = "\n".join(records.astype('str'))
            f.write(records + "\n")

    def print(self, sess, model_file_machine="R"):
        for ly in self.layers:
            if isinstance(ly, Dense):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(model_file_machine)
                    print(sess.run(weight))
