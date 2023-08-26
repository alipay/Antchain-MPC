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
from stensorflow.basic.operator.inverse_sqrt import careful_inverse_sqrt
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
                # print("ly=", ly)
                # print("ly.ploss_pw=", ly.ploss_pw)
                # print("ly.w=", ly.w)
                for i in range(len(ly.w)):
                    wi = ly.w[i]
                    assert isinstance(wi, SharedVariablePair) or isinstance(wi, PrivateVariable)
                    ploss_pwi = ly.ploss_pw[i] + l2_regularization * wi
                    if momentum > 0.0:
                        v = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR,
                                               shape=ploss_pwi.zeros_like())
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
        """

        :param learning_rate:
        :param batch_num:
        :param l2_regularization:
        :param sess:
        :param momentum:
        """
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
                if i % 10 == 0:
                    print("time=", time.time() - start_time)
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

    def get_train_adam_op(self, learningRate=1e-3, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=0.0, amsgrad=False):
        """
        Construct a new Adam optimizer
        """
        if not 0.0 <= learningRate:
            raise ValueError("Invalid learning rate: {}".format(learningRate))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if amsgrad:
            raise NotImplementedError
        if weight_decay > 0:
            raise NotImplementedError
        train_ops1 = []
        for ly in self.layers:
            if isinstance(ly, Layer):
                for i in range(len(ly.w)):
                    wi = ly.w[i]
                    assert isinstance(wi, SharedVariablePair) or isinstance(wi, PrivateVariable)
                    # https://arxiv.org/pdf/1412.6980.pdf
                    ploss_pwi = ly.ploss_pw[i]
                    beta1 = betas[0]
                    beta2 = betas[1]

                    m = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR,
                                           shape=ploss_pwi.shape)
                    m.load_from_numpy(np.zeros(shape=ploss_pwi.shape))

                    v = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR,
                                           shape=ploss_pwi.shape)
                    v.load_from_numpy(np.zeros(shape=ploss_pwi.shape))

                    v_hat = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR,
                                               shape=ploss_pwi.shape)
                    v_hat.load_from_numpy(np.zeros(shape=ploss_pwi.shape))
                    beta1_power_t = tf.Variable(initial_value=tf.ones_like(beta1, dtype='float64'), dtype='float64')
                    beta2_power_t = tf.Variable(initial_value=tf.ones_like(beta2, dtype='float64'), dtype='float64')
                    inv = SharedVariablePair(ownerL=ploss_pwi.ownerL, ownerR=ploss_pwi.ownerR,
                                             shape=ploss_pwi.shape)
                    inv.load_from_tf_tensor(tf.ones(ploss_pwi.shape) / eps)
                    m_new = beta1 * m + (1 - beta1) * ploss_pwi
                    v_new = beta2 * v + (1 - beta2) * ploss_pwi * ploss_pwi
                    beta1_power_t_new = beta1 * beta1_power_t
                    beta2_power_t_new = beta2 * beta2_power_t
                    v_hat_new = v_new / (1 - beta2_power_t_new)
                    step_size = learningRate / (1 - beta1_power_t_new)

                    inv_new1 = careful_inverse_sqrt(v_hat_new, eps)
                    # inv_new1 = invers_sqrt(v_hat_new, eps=eps)
                    wi_new = wi - (step_size * inv) * m_new
                    wi_up_op = wi.assign(wi_new)

                    with tf.control_dependencies([wi_up_op]):
                        m_up_op = m.assign(m_new)
                        v_up_op = v.assign(v_new)
                        beta1_power_t_up_op = beta1_power_t.assign(beta1_power_t_new)
                        beta2_power_t_up_op = beta2_power_t.assign(beta2_power_t_new)
                        v_hat_up_op = v_hat.assign(v_hat_new)
                        inv_up_op1 = inv.assign(inv_new1)
                        # inv_up_op2 = inv.assign(inv_new2)

                    train_ops1 += [wi_up_op,
                                   m_up_op, v_up_op, v_hat_up_op, beta1_power_t_up_op,
                                   beta2_power_t_up_op, inv_up_op1,
                                   tf.print("wi, plosspwi, m_new, v_new, step_size, ins, wi_new=",
                                            [wi.to_tf_tensor("R"), ploss_pwi.to_tf_tensor("R"),
                                             m_new.to_tf_tensor("R"), v_new.to_tf_tensor("R"),
                                             step_size, inv.to_tf_tensor("R"), wi_new.to_tf_tensor("R")])]
        return tf.group(train_ops1)

    def train_adam(self, sess, batch_num, learningRate=1e-3, betas=(0.9, 0.999), eps=1.0 / (1 << 12),
                   weight_decay=0.0, amsgrad=False):
        """

        :param sess:
        :param batch_num:
        :param learningRate:
        :param betas:
        :param eps:
        :param weight_decay:
        :param amsgrad:
        """
        train_op1 = self.get_train_adam_op(learningRate, betas, eps,
                                           weight_decay, amsgrad)
        sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        for i in range(0, batch_num):
            print("batch ", i)
            sess.run(train_op1)
            if i % 10 == 0:
                print("time=", time.time() - start_time)

    def cut_off(self):
        for ly in self.layers:
            assert isinstance(ly, Layer)
            ly.cut_off()

    def predict(self, x):
        raise NotImplementedError
