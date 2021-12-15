#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : NN
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 14:29
   Description : description what the main function of this file
"""
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.basic.basic_class.private import PrivateVariable
from stensorflow.ml.nn.layers.input import Input
import tensorflow as tf
import time
import numpy as np


class NN:
    """
    The class of neural network.
    """
    def __init__(self):
        self.layers = []

    def addLayer(self, ly: Layer):
        # 逐层添加
        # if fathers is not None:
        #     l.fathers = fathers
        for father in ly.fathers:

            if father not in self.layers:
                raise Exception("must add its fathers befor add it to network")
        self.layers += [ly]

    def compile(self):
        # 编译模型
        for ly in self.layers:
            for father in ly.fathers:
                if isinstance(father, Layer):
                    father.add_child(ly)
                else:
                    raise Exception("father must be a layer")
        l_last = self.layers[-1]
        assert isinstance(l_last, Layer)
        l_last.forward()

        for ly in self.layers:
            if isinstance(ly, Input):
                ly.backward()

    def get_train_sgd_op(self, learningRate, l2_regularization, momentum=0.0):
        """
        Construct a new Stochastic Gradient Descent
        """
        train_ops = []
        for ly in self.layers:
            if isinstance(ly, Layer):
                for i in range(len(ly.w)):
                    wi = ly.w[i]
                    assert isinstance(wi, SharedVariablePair) or isinstance(wi, PrivateVariable)
                    ploss_pwi = ly.ploss_pw[i] + l2_regularization * wi
                    if momentum > 0.0:
                        v = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR, shape=ploss_pwi.zeros_like())
                        v.load_from_numpy(np.zeros(shape=ploss_pwi.shape))
                        v_new = momentum * v + ploss_pwi
                        v_up_op = v.assign(v_new)
                        assign_op = wi.assign(wi - learningRate * v_new)
                        train_ops += [v_up_op, assign_op]
                    else:
                        assign_op = wi.assign(wi - learningRate * ploss_pwi)
                        train_ops += [assign_op]
        return tf.group(train_ops)

    def train_sgd(self, learning_rate, batch_num, l2_regularization, sess, momentum=0.0):
        learning_rate_list = None
        # 如果传入学习率list，转换为Tensor
        if isinstance(learning_rate, list):
            learning_rate_list = learning_rate
            self.learning_rate = tf.compat.v1.placeholder(dtype='float64', shape=[])

        else:
            self.learning_rate = learning_rate

        train_op = self.get_train_sgd_op(self.learning_rate, l2_regularization, momentum)
        sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        if learning_rate_list is not None:
            for i in range(batch_num):
                # 每次传入不同的学习率进行梯度下降
                print("batch ", i)
                sess.run(train_op, feed_dict={self.learning_rate: learning_rate_list[i]})
                if i%10==0:
                    print("time=", time.time()-start_time)
        else:
            for i in range(batch_num):
                print("batch ", i)
                sess.run(train_op)
                if i % 10 == 0:
                    print("time=", time.time() - start_time)


                # 输出某层结果的测试代码
                # print(self.layers[3])
                # tmp = sess.run([train_op, self.layers[3].y.to_tf_tensor("R")])
                # print(tmp[1])

    def cut_off(self):
        for ly in self.layers:
            assert isinstance(ly, Layer)
            ly.cut_off()

    def predict(self, x):
        raise NotImplementedError
