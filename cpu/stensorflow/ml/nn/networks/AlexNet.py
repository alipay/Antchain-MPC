#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : AlexNet
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/10/24 下午10:02
   Description : follow  S&P21 https://arxiv.org/pdf/2104.10949.pdf
"""


from stensorflow.ml.nn.networks.NN import NN
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.relu import ReLU
from stensorflow.ml.nn.layers.pooling import AveragePooling2D, MaxPooling2D
from stensorflow.ml.nn.layers.conv2d import Conv2d
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.ml.nn.layers.loss import CrossEntropyLossWithSoftmax
from stensorflow.ml.nn.layers.flatten import Flatten
from stensorflow.ml.nn.layers.dense import Dense
from stensorflow.random import random
import numpy as np





class AlexNet(NN):
    """4 layer CNN.  NetworkC in SecureNN """
    def __init__(self, feature, label, pooling='avg', dataset='cifar10', input_shape=[32, 32, 3]):
        super(AlexNet, self).__init__()
        # input layer, init data；
        layer = Input(dim=feature.shape[1:], x=feature)
        self.addLayer(layer)

        # tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu'),
        layer = Conv2d(filters=96, kernel_size=11, fathers=[layer], strides=[1,4,4,1], padding="SAME", input_shape=input_shape, activate='relu')
        self.addLayer(layer)

        # tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2)),
        if pooling == 'avg':
            layer = AveragePooling2D(output_dim=None, fathers=[layer], pool_size=(3, 3), strides=(2, 2))
        else:
            raise NotImplementedError
        self.addLayer(layer)

        #
        # tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layer = Conv2d(filters=256, kernel_size=5, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)



        # tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2)),
        if pooling == 'avg':
            layer = AveragePooling2D(output_dim=None, fathers=[layer], pool_size=(3, 3), strides=(2, 2))
        else:
            raise NotImplementedError
        self.addLayer(layer)


        #
        # tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        layer = Conv2d(filters=384, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)


        # tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        layer = Conv2d(filters=384, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)


        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layer = Conv2d(filters=256, kernel_size=3, fathers=[layer], padding="SAME", input_shape=layer.output_dim, activate='relu')
        self.addLayer(layer)


        if dataset == 'cifar10':
            # tf.keras.layers.Flatten(),
            # flatten data, only consider data_format = "NWHC"
            layer = Flatten(output_dim=None, fathers=[layer])
            self.addLayer(layer)

            # tf.keras.layers.Dense(256, activation='relu'),
            layer = Dense(output_dim=256, fathers=[layer], activate='relu')
            self.addLayer(layer)

            # # Final Layer
            # tf.keras.layers.Dense(256, activation='relu'),
            layer = Dense(output_dim=256, fathers=[layer], activate='relu')
            self.addLayer(layer)

            # tf.keras.layers.Dense(10, name="Dense"),
            #tf.keras.layers.Activation('softmax')
            layer = Dense(output_dim=10, fathers=[layer], activate='relu')
            self.addLayer(layer)

            # 输出层
            layer_label = Input(dim=10, x=label)
            self.addLayer(ly=layer_label)
        elif dataset == 'TinyImageNet':
            if pooling == 'avg':
                layer = AveragePooling2D(output_dim=None, fathers=[layer], pool_size=(2, 2), strides=(2, 2))
            else:
                raise NotImplementedError
            self.addLayer(layer)

            layer = Flatten(output_dim=None, fathers=[layer])
            self.addLayer(layer)

            layer = Dense(output_dim=1024, fathers=[layer], activate='relu')
            self.addLayer(layer)

            layer = Dense(output_dim=1024, fathers=[layer], activate='relu')
            self.addLayer(layer)

            layer = Dense(output_dim=200, fathers=[layer])
            self.addLayer(layer)

            # 输出层
            layer_label = Input(dim=200, x=label)
            self.addLayer(ly=layer_label)

        else:
            raise Exception("Unknown dataset")




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

    def predict_to_file(self, sess, x, predict_file_name,
                        pred_batch_num, model_file_machine,
                        out_prob=True):
        """
        Save prediction results to file
        Computation is done in batches.
        :param sess: session
        :param x: PrivateTensor. Input samples.
        :param predict_file_name: String.
        :param pred_batch_num: Number of samples per batch.
        :param model_file_machine: String.
        :param record_num_ceil_mod_batch_size:
        :param with_sigmoid:
        :return:
        """
        y_pred = self.predict(x=x,  out_prob=out_prob)
        id_y_pred = y_pred.to_tf_str(owner=model_file_machine)
        random.random_init(sess)
        # 分批写入文件
        with open(predict_file_name, "w") as f:
            for batch in range(pred_batch_num):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                # records.to_file()
                f.write(records + "\n")

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

    def replace_weight(self, model_weight):
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
                kernel.load_from_tf_tensor(model_weight[i])
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
                kernel1.load_from_tf_tensor(model_weight[i])
                kernel2 = ly.w[1]
                assert isinstance(kernel2, SharedVariablePair)
                kernel2.load_from_tf_tensor(model_weight[i + 1])
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



    def load_model(self, save_file_path):
        """
        Save the model weights to npz file.
        :param sess: session
        :param save_file_path: String, path to save the model.
        :param model_file_machine: String, the machine name.
        :return:
        """
        res = np.load(save_file_path)
        self.replace_weight(res)
