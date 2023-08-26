#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group Copyright (c) 2021
   All Rights Reserved.
"""
from stensorflow.ml.nn.networks.NN import NN
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.relu import *
from stensorflow.ml.nn.layers.conv2d import Conv2dLocal
from stensorflow.ml.nn.layers.pooling import avg_pool2d, sum_pool2d_grad
from stensorflow.ml.nn.layers.flatten import *
from stensorflow.ml.nn.layers.dense import *
from stensorflow.ml.nn.layers.loss import *
from stensorflow.random import random


class LocalCNN(NN):
    """
    只有一层卷积核一层池化的CNN网络
    """
    def __init__(self, feature: PrivateTensor, label: Union[PrivateTensor, SharedPair],  loss=None):
        super(LocalCNN, self).__init__()
        # input layer, init data；
        # 这里将dim设置位输入的wight,后续不使用；仅仅是为了应用原有的模板
        layer = Input(dim=28, x=feature)
        local_layer_owner = layer.owner
        self.addLayer(ly=layer)
        # convolutional layer with 1 input channel, 16 output channels and a 5×5 filter
        layer = Conv2dLocal(output_dim=None, fathers=[layer], filters=16,
                            kernel_size=5, input_shape=[28, 28, 1], owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer
        layer = ReLU_Local(output_dim=layer.output_dim, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Average pool
        layer = AveragePooling2DLocal(output_dim=None, fathers=[layer], pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        # flatten data, only consider data_format = "NWHC"
        # 这里需要给出正确的output_dim，方便后续的全连接层
        layer = FlattenLocal(output_dim=None, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 全连接层
        # 这里添加一层，a 2304 × 100 linear layer
        # Dlayer = Dense_Local(output_dim=100, fathers=[layer], owner=local_layer_owner)
        # self.addLayer(Dlayer)
        # 添加一层Relu
        # Relu Layer
        # layer = ReLU_Local(output_dim=100, fathers=[Dlayer], owner=local_layer_owner)
        # self.addLayer(layer)
        # a 2304 × 10 linear layer； a 100* 10 line layer
        layer = Dense_Local(output_dim=10, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失计算
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
                        record_num_ceil_mod_batch_size,
                        with_sigmoid):
        y_pred = self.predict(x=x,  out_prob=with_sigmoid)
        id_y_pred = y_pred.to_tf_str(owner=model_file_machine)
        random.random_init(sess)
        # 分批写入文件
        with open(predict_file_name, "w") as f:
            for batch in range(pred_batch_num):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                # records.to_file()
                f.write(records + "\n")

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


class LocalNetworkB(NN):
    """
    两层卷积和两层池化的复杂网络
    """
    def __init__(self, feature: PrivateTensor, label: Union[PrivateTensor, SharedPair], loss=None):
        super(LocalNetworkB, self).__init__()
        # 这里将dim设置位输入的wight,后续不使用；仅仅是为了应用原有的模板
        layer = Input(dim=28, x=feature)
        local_layer_owner = layer.owner
        self.addLayer(layer)
        # convolutional layer with 1 input channel, 16 output channels and a 5×5 filter
        layer = Conv2dLocal(output_dim=None, fathers=[layer], filters=16,
                            kernel_size=5, input_shape=[28, 28, 1], owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer
        layer = ReLU_Local(output_dim=layer.output_dim, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Average pool
        layer = AveragePooling2DLocal(output_dim=None, fathers=[layer],
                                      pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        # 16 input channels, 16 output channels and another 5×5 filter
        layer = Conv2dLocal(output_dim=None, fathers=[layer], filters=16,
                            kernel_size=5, input_shape=layer.output_dim, owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer
        layer = ReLU_Local(output_dim=layer.output_dim, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Average pool
        layer = AveragePooling2DLocal(output_dim=None, fathers=[layer], pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        # flatten data, only consider data_format = "NWHC"
        # 这里需要给出正确的output_dim，方便后续的全连接层
        layer = FlattenLocal(output_dim=None, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 全连接层
        # 256×100 fully-connected layer
        layer = Dense_Local(output_dim=100, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer, 需要给出一个正确的output_dim
        layer = ReLU_Local(output_dim=100, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # a 100 × 10 linear layer
        layer = Dense_Local(output_dim=10, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失计算
        layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label,
                                                      owner=local_layer_owner)
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
                        record_num_ceil_mod_batch_size,
                        with_sigmoid):
        y_pred = self.predict(x=x,  out_prob=with_sigmoid)
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


class LocalNetworkC(NN):
    """
    两层卷积和两层池化的复杂网络
    """
    def __init__(self, feature: PrivateTensor, label: Union[PrivateTensor, SharedPair], loss=None):
        super(LocalNetworkC, self).__init__()
        # 这里将dim设置位输入的wight,后续不使用；仅仅是为了应用原有的模板
        layer = Input(dim=28, x=feature)
        local_layer_owner = layer.owner
        self.addLayer(layer)
        # convolutional layer with 1 input channel, 16 output channels and a 5×5 filter
        layer = Conv2dLocal(output_dim=None, fathers=[layer], filters=20,
                            kernel_size=5, input_shape=[28, 28, 1], owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer
        layer = ReLU_Local(output_dim=layer.output_dim, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Average pool
        layer = AveragePooling2DLocal(output_dim=None, fathers=[layer],
                                      pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)


        # 20 input channels, 50 output channels and another 5×5 filter
        layer = Conv2dLocal(output_dim=None, fathers=[layer], filters=50,
                            kernel_size=5, input_shape=layer.output_dim, owner=local_layer_owner)
        self.addLayer(layer)
        # Relu Layer
        layer = ReLU_Local(output_dim=layer.output_dim, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # Average pool
        layer = AveragePooling2DLocal(output_dim=None, fathers=[layer], pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)
        # flatten data, only consider data_format = "NWHC"
        # 这里需要给出正确的output_dim，方便后续的全连接层
        layer = FlattenLocal(output_dim=None, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 全连接层
        # 800x500 fully-connected layer
        layer = Dense_Local(output_dim=500, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # a 500 × 10 linear layer
        layer = Dense_Local(output_dim=10, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失计算
        layer_loss = CrossEntropyLossWithSoftmaxLocal(layer_score=layer, layer_label=layer_label,
                                                      owner=local_layer_owner)
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
                        with_sigmoid):
        y_pred = self.predict(x=x,  out_prob=with_sigmoid)
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

