#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : GCN.py
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/4/22 下午4:59
   Description : description what the main function of this file
"""
from stensorflow.ml.nn.layers.layer import Layer
import tensorflow as tf
from stensorflow.ml.nn.networks.NN import NN
from typing import List
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.graphconv import GraphConv
from stensorflow.ml.nn.layers.loss import Loss, BinaryCrossEntropyLossWithSigmoid, \
    CrossEntropyLossWithSoftmax, MSE
from stensorflow.ml.nn.layers.relu import ReLU
from stensorflow.random.random import random_init

class GCN(NN):
    def __init__(self, feature: PrivateTensor, label: PrivateTensor, dense_dims: List[int],
                 adjacency_matrix: PrivateTensor, train_mask, loss=None):
        """
        Graph Convalution Network.
        :param feature:
        :param label:
        :param dense_dims:  [d0, d1, ..., dn]  where d0=dim(feature) dn=dim(label), di is the
        dimension of i-th hidden layer
        :param loss
        """

        if len(dense_dims) < 2:
            raise Exception("must have len(dense_dims)>=2")

        if feature.shape[1] != dense_dims[0]:
            raise Exception("must have x.shape[1] == dense_dims[0]")
        super(GCN, self).__init__()
        layer = Input(dim=dense_dims[0], x=feature)
        self.addLayer(ly=layer)
        self.adjacency_matrix = adjacency_matrix
        self.train_mask = np.reshape(train_mask.astype(np.int64), [-1,1])
        self.test_mask = 1 - self.train_mask
        input_layers = [layer]

        for i in range(1, len(dense_dims)):
            if i == 1:
                layer = GraphConv(output_dim=dense_dims[i], fathers=input_layers,
                                  adjacency_matrix=self.adjacency_matrix, train_mask=self.train_mask)
            else:
                layer = GraphConv(output_dim=dense_dims[i], fathers=[layer],
                                  adjacency_matrix=self.adjacency_matrix, train_mask=self.train_mask)
            self.addLayer(ly=layer)
            if i < len(dense_dims) - 1:
                layer = ReLU(output_dim=dense_dims[i], fathers=[layer])
                self.addLayer(ly=layer)

        layer_label = Input(dim=dense_dims[-1], x=label)
        self.addLayer(ly=layer_label)

        if loss is None or loss == "BinaryCrossEntropyLossWithSigmoid" or \
                loss == Loss.BinaryCrossEntropyLossWithSigmoid:
            layer_loss = BinaryCrossEntropyLossWithSigmoid(layer_score=layer,
                                                           layer_label=layer_label,
                                                           train_mask=self.train_mask)
            self.addLayer(ly=layer_loss)
        elif loss == "CrossEntropyLossWithSoftmax" or loss == Loss.CrossEntropyLossWithSoftmax:
            layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label,
                                                     train_mask=self.train_mask)
            self.addLayer(ly=layer_loss)
        elif loss == "MSE" or loss == Loss.MSE:
            layer_loss = MSE(layer_score=layer, layer_label=layer_label)
            self.addLayer(ly=layer_loss)
        else:
            raise Exception("unsupposed loss")

    def predict(self, out_prob=True) -> SharedPair:
        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        if out_prob:
            return ly.y
        else:
            return ly.score


    def print(self, sess, model_file_machine="R"):
        for ly in self.layers:
            if isinstance(ly, GraphConv):
                for weight in ly.w:
                    weight_tf = weight.to_tf_tensor(owner=model_file_machine)
                    print(sess.run(weight_tf))

    def save(self, sess, model_file_machine="R", path="./output/model"):
        print("save model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, GraphConv):
                ly.save(model_file_machine, sess, path + "/param_{}".format(i))
            i += 1

    def load(self, path="./output/model/"):
        print("load model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, GraphConv):
                ly.load(path + "/param_{}".format(i))
            i += 1


    def predict_to_file(self, sess, predict_file_name,
                        idx, model_file_machine="R",
                        with_sigmoid=True, single_out=False):
        print("predict_file_name=", predict_file_name)
        y_pred = self.predict(out_prob=with_sigmoid)
        if not single_out:
            id_y_pred = y_pred.to_tf_str(owner=model_file_machine, id_col=idx)
        else:
            y_pred = tf.argmax(y_pred.to_tf_tensor(owner=model_file_machine), axis=1)
            y_pred = tf.reshape(y_pred, [-1, 1])
            y_pred = tf.strings.as_string(y_pred)
            if idx is not None:
                id_y_pred = tf.concat([idx, y_pred], axis=1)
                id_y_pred = tf.compat.v1.reduce_join(id_y_pred, separator=",", axis=-1)
        random_init(sess)

        with open(predict_file_name, "w") as f:
            records = sess.run(id_y_pred)
            records = "\n".join(records.astype('str'))
            f.write(records + "\n")


    def print(self, sess, model_file_machine="R"):
        for ly in self.layers:
            if isinstance(ly, GraphConv):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(model_file_machine)
                    print(sess.run(weight))
