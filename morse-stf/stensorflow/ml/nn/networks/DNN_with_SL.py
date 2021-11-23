#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : DNN_with_SL
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-12-17 11:27
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.exception.exception import StfValueException
from stensorflow.ml.nn.networks.NN import NN
import tensorflow as tf
from typing import Union, List
from stensorflow.random.random import random_init
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.dense import Dense, Dense_Local
from stensorflow.ml.nn.layers.sum import Sum
from stensorflow.ml.nn.layers.loss import Loss, BinaryCrossEntropyLossWithSigmoid, \
    BinaryCrossEntropyLossWithSigmoidLocal, CrossEntropyLossWithSoftmax, CrossEntropyLossWithSoftmaxLocal
from stensorflow.ml.nn.layers.relu import ReLU, ReLU_Local


class DNN_with_SL(NN):
    def __init__(self, feature: PrivateTensor, label: Union[PrivateTensor, SharedPair], dense_dims: List[int],
                 secure_config: list = None, loss=None):
        """
        单方特征DNN
        :param feature:
        :param label:
        :param dense_dims: [d0, d1, ..., dn]  where d0=dim(feature) dn=dim(output)
        :param secure_config:  [n0, n1, n2]
        where n0>=1, n1,n2>=0 and n0+n1+n2=n
        it meanse secure_levels=["local"]xn0+["share"]xn1+["local"]xn2
        """

        if len(dense_dims) < 2:
            raise Exception("must have len(dense_dims)>=2")
        if secure_config is None:
            secure_config = [1, len(dense_dims) - 1, 0]

        if not isinstance(secure_config, list):
            raise Exception("secure_config must be a list of int")
        elif len(secure_config) != 3:
            raise Exception("must have len(secure_config)==3")
        elif not (
                isinstance(secure_config[0], int) and isinstance(secure_config[1], int) and isinstance(secure_config[2],
                                                                                                       int)):
            raise Exception(
                "must have isinstance(secure_config[0], int) and isinstance(secure_config[1], int) and isinstance("
                "secure_config[2], int)")
        elif secure_config[0] < 1 or secure_config[1] < 0 or secure_config[2] < 0 or sum(secure_config) != len(
                dense_dims):
            raise Exception(
                "must have secure_config[0]>=1 and secure_config[1]>=0 and secure_config[2]>=0 and sum("
                "secure_config)==len(dense_dims)")
        else:
            pass

        super(DNN_with_SL, self).__init__()
        self.secure_config = secure_config
        layer = Input(dim=feature.shape[1], x=feature)

        local_layer_owner = layer.owner
        self.addLayer(ly=layer)

        for i in range(1, secure_config[0]):
            layer = Dense_Local(output_dim=dense_dims[i], fathers=[layer], owner=local_layer_owner, with_b=True)

            self.addLayer(ly=layer)
            if i < len(dense_dims) - 1:
                layer = ReLU_Local(output_dim=dense_dims[i], fathers=[layer], owner=local_layer_owner)

                self.addLayer(ly=layer)

        for i in range(secure_config[0], secure_config[0] + secure_config[1]):
            layer = Dense(output_dim=dense_dims[i], fathers=[layer])

            self.addLayer(ly=layer)
            if i < len(dense_dims) - 1:
                layer = ReLU(output_dim=dense_dims[i], fathers=[layer])

                self.addLayer(ly=layer)

        for i in range(secure_config[0] + secure_config[1], sum(secure_config)):
            layer = Dense_Local(output_dim=dense_dims[i], fathers=[layer], owner=label.owner, with_b=True)

            self.addLayer(ly=layer)
            if i < len(dense_dims) - 1:
                layer = ReLU_Local(output_dim=dense_dims[i], fathers=[layer], owner=label.owner)

                self.addLayer(ly=layer)

        # layer_label = Input(dim=1, x=label)
        layer_label = Input(dim=dense_dims[-1], x=label)
        self.addLayer(ly=layer_label)

        if loss is None:
            loss = "BinaryCrossEntropyLossWithSigmoid"

        if loss == "BinaryCrossEntropyLossWithSigmoid" or loss == Loss.BinaryCrossEntropyLossWithSigmoid:
            if secure_config[-1] == 0:
                layer_loss = BinaryCrossEntropyLossWithSigmoid(layer_score=layer, layer_label=layer_label)
            else:
                layer_loss = BinaryCrossEntropyLossWithSigmoidLocal(layer_score=layer, layer_label=layer_label,
                                                                    owner=layer_label.owner)
        elif loss == "CrossEntropyLossWithSoftmax" or loss == Loss.CrossEntropyLossWithSoftmax:
            if secure_config[-1] == 0:
                layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
            else:
                layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label,
                                                              owner=layer_label.owner)
        else:
            raise StfValueException("loss", "BinaryCrossEntropyLossWithSigmoid or CrossEntropyLossWithSoftmax", loss)

        self.addLayer(ly=layer_loss)

    def predict(self, x, x_another=None, out_prob=True) -> Union[SharedPair, PrivateTensor]:
        self.cut_off()
        l_input = self.layers[0]
        if not isinstance(l_input, Input):
            raise Exception("l_input mast be a Input layer")
        l_input.replace(x)
        self.layers[0] = l_input

        if x_another is not None:
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
                        batch_num, idx, model_file_machine, record_num_ceil_mod_batch_size,
                        x_another=None, with_sigmoid=True):
        y_pred = self.predict(x=x, x_another=x_another, out_prob=with_sigmoid)

        id_y_pred = y_pred.to_tf_str(owner=model_file_machine, id_col=idx)
        sess.run(tf.compat.v1.global_variables_initializer())
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
            if isinstance(ly, Dense) or isinstance(ly, Dense_Local):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    sess.run(weight)

    def save(self, sess, model_file_machine="R", path="./output/model"):
        print("save model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Dense_Local):
                ly.save(model_file_machine, sess, path + "/param_{}".format(i))
            i += 1

    def load(self, path="./output/model/"):
        print("load model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Dense_Local):
                ly.load(path + "/param_{}".format(i))
            i += 1


class DNN_with_SL2(NN):
    def __init__(self, feature: PrivateTensor, feature_another: PrivateTensor, label: PrivateTensor,
                 dense_dims: List[int], secure_config: list = None, loss=None):
        """

        :param feature:
        :param label:
        :param dense_dims: [d0, d1, ..., dn]  where d0=dim(feature) dn=dim(output)
        :param feature_another:
        :param secure_config:  [n0, n1, n2]
        where n0>=1, n1,n2>=0 and n0+n1+n2=n+1
        it meanse secure_levels=["local"]xn0+["share"]xn1+["local"]xn2
        """

        if len(dense_dims) < 2:
            raise Exception("must have len(dense_dims)>=2")
        if secure_config is None:
            secure_config = [1, len(dense_dims) - 1, 0]

        if not isinstance(secure_config, list):
            raise Exception("secure_config must be a list of int")
        elif len(secure_config) != 3:
            raise Exception("must have len(secure_config)==3")
        elif not (
                isinstance(secure_config[0], int) and isinstance(secure_config[1], int) and isinstance(secure_config[2],
                                                                                                       int)):
            raise Exception(
                "must have isinstance(secure_config[0], int) and isinstance(secure_config[1], int) and isinstance("
                "secure_config[2], int)")
        elif secure_config[0] < 1 or secure_config[1] < 0 or secure_config[2] < 0 or sum(secure_config) != len(
                dense_dims):
            raise Exception(
                "must have secure_config[0]>=1 and secure_config[1]>=0 and secure_config[2]>=0 and sum("
                "secure_config)==len(dense_dims)")
        else:
            pass

        super(DNN_with_SL2, self).__init__()
        self.secure_config = secure_config
        layer = Input(dim=feature.shape[1], x=feature)
        self.addLayer(ly=layer)

        if feature.shape[1] + feature_another.shape[1] != dense_dims[0]:
            raise Exception("must have feature.shape[1]+feature_another.shape[1] == dense_dims[0]")
        if feature.owner == feature_another.owner:
            raise Exception("must have feature.owner != feature_another.owner")
        layer_another = Input(dim=feature_another.shape[1], x=feature_another)
        self.addLayer(layer_another)

        if secure_config[0] == 1:  # local, share, .....
            layer = Dense(output_dim=dense_dims[1], fathers=[layer, layer_another])
            self.addLayer(ly=layer)
        elif secure_config[0] == 2:  # local, local, share, ...
            layer = Dense_Local(output_dim=dense_dims[1], fathers=[layer], owner=layer.owner, with_b=True)
            self.addLayer(ly=layer)
            layer_another = Dense_Local(output_dim=dense_dims[1], fathers=[layer_another], owner=layer_another.owner,
                                        with_b=False)
            self.addLayer(ly=layer_another)
            layer = Sum(output_dim=dense_dims[1], fathers=[layer, layer_another])
            self.addLayer(ly=layer)
        else:
            raise Exception("must have secure_config[0]==1 or 2")
        if len(dense_dims) > 2:
            layer = ReLU(output_dim=dense_dims[1], fathers=[layer])
            self.addLayer(ly=layer)

        for i in range(2, secure_config[0] + secure_config[1]):
            layer = Dense(output_dim=dense_dims[i], fathers=[layer])
            self.addLayer(ly=layer)
            if len(dense_dims) > 1 + i:
                layer = ReLU(output_dim=dense_dims[i], fathers=[layer])
                self.addLayer(ly=layer)

        for i in range(secure_config[0] + secure_config[1], sum(secure_config)):
            layer = Dense_Local(output_dim=dense_dims[i], fathers=[layer], owner=label.owner, with_b=True)
            self.addLayer(ly=layer)
            if len(dense_dims) > 1 + i:
                layer = ReLU_Local(output_dim=dense_dims[i], fathers=[layer], owner=label.owner)
                self.addLayer(ly=layer)

        layer_label = Input(dim=1, x=label)
        self.addLayer(ly=layer_label)

        if loss is None:
            loss = "BinaryCrossEntropyLossWithSigmoid"

        if loss == "BinaryCrossEntropyLossWithSigmoid" or loss == Loss.BinaryCrossEntropyLossWithSigmoid:
            if secure_config[-1] == 0:
                layer_loss = BinaryCrossEntropyLossWithSigmoid(layer_score=layer, layer_label=layer_label)
            else:
                layer_loss = BinaryCrossEntropyLossWithSigmoidLocal(layer_score=layer, layer_label=layer_label,
                                                                    owner=layer_label.owner)
        elif loss == "CrossEntropyLossWithSoftmax" or loss == Loss.CrossEntropyLossWithSoftmax:
            if secure_config[-1] == 0:
                layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
            else:
                layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label,
                                                              owner=layer_label.owner)
        else:
            raise StfValueException("loss", "BinaryCrossEntropyLossWithSigmoid or CrossEntropyLossWithSoftmax", loss)

        self.addLayer(ly=layer_loss)

    def predict(self, x, x_another=None, with_sigmoid=True) -> Union[SharedPair, PrivateTensor]:
        self.cut_off()
        l_input = self.layers[0]
        if not isinstance(l_input, Input):
            raise Exception("l_input mast be a Input layer")
        l_input.replace(x)
        # self.layers[0] = l_input

        if x_another is not None:
            l_input_another = self.layers[1]
            if not isinstance(l_input_another, Input):
                raise Exception("l_input_another mast be a Input layer")
            l_input_another.replace(x_another)
            # self.layers[1] = l_input_another

        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        else:
            ly.forward()

        if with_sigmoid:
            return ly.y
        else:
            return ly.score

    def predict_to_file(self, sess, x, predict_file_name,
                        batch_num, idx, model_file_machine, record_num_ceil_mod_batch_size,
                        x_another=None, with_sigmoid=True):
        y_pred = self.predict(x=x, x_another=x_another, with_sigmoid=with_sigmoid)

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
            if isinstance(ly, Dense) or isinstance(ly, Dense_Local):
                for weight in ly.w:
                    weight_tf = weight.to_tf_tensor(owner=model_file_machine)
                    print(sess.run(weight_tf))

    def save(self, sess, model_file_machine="R", path="./output/model"):
        print("save model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, (Dense, Dense_Local)):
                ly.save(model_file_machine, sess, path + "/param_{}".format(i))
            i += 1

    def load(self, path="./output/model/"):
        print("load model...")
        i = 0
        for ly in self.layers:
            if isinstance(ly, (Dense, Dense_Local)):
                ly.load(path + "/param_{}".format(i))
            i += 1
