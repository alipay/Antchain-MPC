#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : AlexNet
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/10/25 下午5:19
   Description :  follow  S&P21 https://arxiv.org/pdf/2104.10949.pdfv
"""

from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from typing import Union
from stensorflow.ml.nn.networks.NN import NN
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.relu import ReLU, ReLU_Local
from stensorflow.ml.nn.layers.pooling import AveragePooling2D, MaxPooling2D, AveragePooling2DLocal, MaxPooling2DLocal
from stensorflow.basic.operator.softmax import softmax
from stensorflow.ml.nn.layers.conv2d import Conv2d, Conv2dLocal
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.ml.nn.layers.loss import CrossEntropyLossWithSoftmax, CrossEntropyLossWithSoftmaxLocal
from stensorflow.ml.nn.layers.flatten import Flatten, FlattenLocal
from stensorflow.ml.nn.layers.dense import Dense, Dense_Local
from stensorflow.random import random
import numpy as np




class VGG16(NN):
    def __init__(self, feature, label, pooling='avg', input_shape=[32, 32, 3], dataset='cifar10'):
        super(VGG16, self).__init__()
        # input layer, init data；
        if pooling == 'avg':
            pooling_layer = AveragePooling2D
        elif pooling == 'max':
            pooling_layer = MaxPooling2D
        else:
            raise NotImplementedError
        layer = Input(dim=feature.shape[1:], x=feature)
        self.addLayer(layer)

        # 1
        layer = Conv2d(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=input_shape, activate='relu')
        self.addLayer(layer)

        # 2
        layer = Conv2d(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
        self.addLayer(layer)

        # 3
        layer = Conv2d(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 4
        layer = Conv2d(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
        self.addLayer(layer)


        # 5
        layer = Conv2d(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 6
        layer = Conv2d(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)


        # 7
        layer = Conv2d(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
        self.addLayer(layer)

        # 8
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 9
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 10
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
        self.addLayer(layer)


        # 11
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 12
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)

        # 13
        layer = Conv2d(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
        self.addLayer(layer)

        if dataset == 'cifar10':
            # 14
            layer = Flatten(output_dim=None, fathers=[layer])
            self.addLayer(layer)

            # 15
            layer = Dense(output_dim=256, fathers=[layer], activate='relu')
            self.addLayer(layer)

            # 16
            layer = Dense(output_dim=256, fathers=[layer], activate='relu')
            self.addLayer(layer)

            layer = Dense(output_dim=10, fathers=[layer])
            self.addLayer(layer)

            # 输出层
            layer_label = Input(dim=10, x=label)
            self.addLayer(ly=layer_label)
        elif dataset == 'TinyImageNet':
            layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
            self.addLayer(layer)

            layer = Flatten(output_dim=None, fathers=[layer])
            self.addLayer(layer)

            layer = Dense(output_dim=512, fathers=[layer], activate='relu')
            self.addLayer(layer)

            layer = Dense(output_dim=512, fathers=[layer], activate='relu')
            self.addLayer(layer)

            layer = Dense(output_dim=200, fathers=[layer], activate='relu')
            self.addLayer(layer)

            # 输出层
            layer_label = Input(dim=200, x=label)
            self.addLayer(ly=layer_label)
        else:
            raise Exception("unknown dataset")
        # 损失层
        layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
        self.addLayer(ly=layer_loss)

    def predict(self, x, out_prob=True):
        """
        Generates output predictions for the input samples.
        :param x: Input samples.
        :param out_prob: True or False
        :return: The predicted result or probability
        """
        self.cut_off()
        # Input Layer
        l_input = self.layers[0]
        assert isinstance(l_input, Input)
        l_input.replace(x)
        self.layers[0] = l_input
        # Output Layer
        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        else:
            ly.forward()

        if out_prob:
            return ly.y
        else:
            return ly.score



    def print(self, model_file_machine, sess):
        """
        print model weights
        """
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Conv2d):
                print(ly)
                for weight in ly.w:
                    print(weight)
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    print("***")
                    print(sess.run(weight))

    def replace_weight(self, keras_weight):
        """
        将传入的权重赋值给当前网络
        """
        i = 0
        for ly in self.layers:
            if isinstance(ly, Conv2d):
                # 用传入的权重直接进行预测
                # kernel = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i].shape)
                # kernel.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel
                # 用传入的权重进行训练
                kernel = ly.w[0]
                assert isinstance(kernel, SharedVariablePair)
                kernel.load_from_tf_tensor(keras_weight[i])
                i += 1

            if isinstance(ly, Dense):
                # 用传入的权重直接进行预测
                # kernel1 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i].shape)
                # kernel1.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel1
                # kernel2 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i+1].shape)
                # kernel2.load_from_numpy(keras_weight[i+1])
                # ly.w[1] = kernel2
                # 用传入的权重进行训练
                kernel1 = ly.w[0]
                assert isinstance(kernel1, SharedVariablePair)
                kernel1.load_from_tf_tensor(keras_weight[i])
                kernel2 = ly.w[1]
                assert isinstance(kernel2, SharedVariablePair)
                kernel2.load_from_tf_tensor(keras_weight[i+1])
                i += 2

    def save_model(self, sess, save_file_path, model_file_machine):
        """
        Save the model weights to npz file.
        :param sess: session
        :param save_file_path: String, path to save the model.
        :param model_file_machine: String, the machine name.
        :return:
        """
        res = []
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Conv2d):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    weight = sess.run(weight)
                    res.append(weight)
        res = np.array(res)
        np.savez(save_file_path, weight=res)





class LocalVGG16(NN):
    """
    两层卷积和两层池化的复杂网络
    """
    def __init__(self, feature, label, pooling='avg'):
        super(LocalVGG16, self).__init__()
        # input layer, init data；
        if pooling == 'avg':
            pooling_layer = AveragePooling2DLocal
        elif pooling == 'max':
            pooling_layer = MaxPooling2DLocal
        else:
            raise NotImplementedError
        layer = Input(dim=feature.shape[1:], x=feature)
        local_layer_owner = layer.owner
        self.addLayer(layer)

        # 1
        layer = Conv2dLocal(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=[32, 32, 3], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 2
        layer = Conv2dLocal(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)

        # 3
        layer = Conv2dLocal(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 4
        layer = Conv2dLocal(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)


        # 5
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 6
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)


        # 7
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)

        # 8
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 9
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 10
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)


        # 11
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 12
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 13
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        #
        # 14
        layer = FlattenLocal(output_dim=None, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        #
        # 15
        layer = Dense_Local(output_dim=256, fathers=[layer], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 16
        layer = Dense_Local(output_dim=256, fathers=[layer], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = Dense_Local(output_dim=10, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)

        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失层
        layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label, owner=local_layer_owner)
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

    def predict_to_file(self, sess, x, predict_file_name,
                        pred_batch_num, model_file_machine,
                        out_prob=True):
        y_pred = self.predict(x=x,  out_prob=out_prob)
        id_y_pred = y_pred.to_tf_str(owner=model_file_machine)
        random.random_init(sess)
        with open(predict_file_name, "w") as f:
            for batch in range(pred_batch_num):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                # records.to_file()
                f.write(records + "\n")

    def save_model(self, sess, save_file_path, model_file_machine):
        res = []
        for ly in self.layers:
            if isinstance(ly, Dense_Local) or isinstance(ly, Conv2dLocal):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    weight = sess.run(weight)
                    res.append(weight)
        res = np.array(res)
        np.savez(save_file_path, weight=res)

    def replace_weight(self, keras_weight):
        i = 0
        for ly in self.layers:
            if isinstance(ly, Conv2dLocal):
                # 用传入的权重直接进行预测
                # kernel = PrivateTensor(owner=ly.owner)
                # kernel.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel
                # 用传入的权重进行训练
                kernel = ly.w[0]
                kernel.load_from_numpy(keras_weight[i])
                i += 1
            if isinstance(ly, Dense_Local):
                # 用传入的权重直接进行预测
                # kernel1 = PrivateTensor(owner=ly.owner)
                # kernel1.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel1
                # kernel2 = PrivateTensor(owner=ly.owner)
                # kernel2.load_from_numpy(keras_weight[i+1])
                # ly.w[1] = kernel2
                # 用传入的权重进行训练
                kernel1 = ly.w[0]
                kernel2 = ly.w[1]
                kernel1.load_from_numpy(keras_weight[i])
                kernel2.load_from_numpy(keras_weight[i+1])
                i += 2



class LocalVGG16_TI(NN):
    """
    两层卷积和两层池化的复杂网络
    """
    def __init__(self, feature, label, pooling='avg'):
        super(LocalVGG16_TI, self).__init__()
        # input layer, init data；
        if pooling == 'avg':
            pooling_layer = AveragePooling2DLocal
        elif pooling == 'max':
            pooling_layer = MaxPooling2DLocal
        else:
            raise NotImplementedError
        layer = Input(dim=feature.shape[1:], x=feature)
        local_layer_owner = layer.owner
        self.addLayer(layer)

        # 1
        layer = Conv2dLocal(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=[64, 64, 3], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 2
        layer = Conv2dLocal(filters=64, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)

        # 3
        layer = Conv2dLocal(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 4
        layer = Conv2dLocal(filters=128, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)


        # 5
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 6
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)


        # 7
        layer = Conv2dLocal(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)

        # 8
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 9
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 10
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)


        # 11
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 12
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 13
        layer = Conv2dLocal(filters=512, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu', owner=local_layer_owner)
        self.addLayer(layer)
        layer = pooling_layer(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        #
        # 14
        layer = FlattenLocal(output_dim=None, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        #
        # 15
        layer = Dense_Local(output_dim=1024, fathers=[layer], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        # 16
        layer = Dense_Local(output_dim=512, fathers=[layer], activate='relu', owner=local_layer_owner)
        self.addLayer(layer)

        layer = Dense_Local(output_dim=200, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)

        # 输出层
        layer_label = Input(dim=200, x=label)
        self.addLayer(ly=layer_label)
        # 损失层
        layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label, owner=local_layer_owner)
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

    def predict_to_file(self, sess, x, predict_file_name,
                        pred_batch_num, model_file_machine,
                        out_prob=True):
        y_pred = self.predict(x=x,  out_prob=out_prob)
        id_y_pred = y_pred.to_tf_str(owner=model_file_machine)
        random.random_init(sess)
        with open(predict_file_name, "w") as f:
            for batch in range(pred_batch_num):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                # records.to_file()
                f.write(records + "\n")

    def save_model(self, sess, save_file_path, model_file_machine):
        res = []
        for ly in self.layers:
            if isinstance(ly, Dense_Local) or isinstance(ly, Conv2dLocal):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    weight = sess.run(weight)
                    res.append(weight)
        res = np.array(res)
        np.savez(save_file_path, weight=res)

    def replace_weight(self, keras_weight):
        i = 0
        for ly in self.layers:
            if isinstance(ly, Conv2dLocal):
                # 用传入的权重直接进行预测
                # kernel = PrivateTensor(owner=ly.owner)
                # kernel.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel
                # 用传入的权重进行训练
                kernel = ly.w[0]
                kernel.load_from_numpy(keras_weight[i])
                i += 1
            if isinstance(ly, Dense_Local):
                # 用传入的权重直接进行预测
                # kernel1 = PrivateTensor(owner=ly.owner)
                # kernel1.load_from_numpy(keras_weight[i])
                # ly.w[0] = kernel1
                # kernel2 = PrivateTensor(owner=ly.owner)
                # kernel2.load_from_numpy(keras_weight[i+1])
                # ly.w[1] = kernel2
                # 用传入的权重进行训练
                kernel1 = ly.w[0]
                kernel2 = ly.w[1]
                kernel1.load_from_numpy(keras_weight[i])
                kernel2.load_from_numpy(keras_weight[i+1])
                i += 2
