#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : logistic_regression
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 11:42
   Description : description what the main function of this file
"""
import tensorflow as tf
import stensorflow as stf
from typing import Union
from stensorflow.basic.basic_class.pair import SharedPair, SharedVariablePair
from stensorflow.exception.exception import StfValueException
from stensorflow.random.random import random_init
from stensorflow.homo_enc.homo_enc import homo_init
from stensorflow.basic.basic_class.base import get_device
from stensorflow.basic.basic_class.private import PrivateTensor
import numpy as np
from stensorflow.global_var import StfConfig
import os
from stensorflow.basic.operator.sigmoid import sigmoid_sin, sigmoid_idea


class LogisticRegression:
    """Logistic regression that features in 1 player."""

    def __init__(self, num_features, learning_rate=0.01, l2_regularization=1E-2):
        """

        :param num_features:  the number of features
        :param learning_rate:  learning rate
        :param l2_regularization: l2 regularization
        """

        self.num_features = num_features
        self.w = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features, 1])
        self.w.load_from_numpy(np.random.uniform(size=[num_features, 1],
                                                 low=-1.0 / np.sqrt(num_features + 1),
                                                 high=1.0 / np.sqrt(num_features + 1)))
        self.b = SharedVariablePair(ownerL="L", ownerR="R", shape=[1])
        self.b.load_from_tf_tensor(tf.zeros([1]))

        self.learning_rate = learning_rate
        self.l2_regularzation = l2_regularization

    @property
    def weights(self):
        return self.w, self.b

    def forward(self, x, with_sigmoid=True):
        """

        :param x: list<Union[PrivateTensor, SharedPair]>
        :param with_sigmoid:
        :return:
        """

        m = x @ self.w
        print("m.xL.shape=", m.xL.shape)
        print("m.xR.shape=", m.xR.shape)
        out = m + self.b

        if with_sigmoid:
            y = sigmoid_sin(out, M=16)
            # y = sigmoid_idea(out, M=16)
        else:
            y = out
        return y

    def backward(self, x: Union[PrivateTensor, SharedPair], dy: Union[PrivateTensor, SharedPair], learning_rate=0.01):
        """
        :param x:    fetures
        :param dy:    ploss/py
        :param learning_rate:   learning rate
        :return:
        """

        batch_size = x.shape[0]
        dw = x.transpose() @ (dy / batch_size) + self.l2_regularzation * self.w
        db = dy.reduce_sum(axis=0) / batch_size

        assign_ops = [
            self.w.assign(self.w - learning_rate * dw),
            self.b.assign(self.b - learning_rate * db),
        ]
        return assign_ops

    def loss_grad(self, y: Union[PrivateTensor, SharedPair], y_hat: Union[PrivateTensor, SharedPair]):
        """

        :param y:
        :param y_hat:
        :return:
        """
        dy = y_hat - y
        return dy

    def fit_batch(self, x: Union[PrivateTensor, SharedPair], y: Union[PrivateTensor, SharedPair]):
        """

        :param x:
        :param y:
        :return:
        """
        y_hat = self.forward(x)
        dy = self.loss_grad(y, y_hat)
        fit_batch_op = self.backward(x, dy, self.learning_rate)
        return fit_batch_op

    def fit(self, sess, x: Union[PrivateTensor, SharedPair], y: Union[PrivateTensor, SharedPair], num_batches,
            progress_file="./train_progress"):
        """

        :param sess:
        :param x:
        :param y:
        :param num_batches:
        :param progress_file:
        """
        fit_batch_op = self.fit_batch(x, y)
        if StfConfig.pre_produce_flag and StfConfig.offline_model:
            pre_produce_op = tf.group(StfConfig.pre_produce_list)
        if sess is not None:
            sess.run(tf.compat.v1.global_variables_initializer())
            with open(progress_file, "a") as f:

                for batch in range(num_batches):
                    print("batch=", batch)
                    if StfConfig.pre_produce_flag and StfConfig.offline_model:
                        sess.run(pre_produce_op)
                    else:
                        sess.run(fit_batch_op)
                    if batch % (int(num_batches / 100) + 1) == 0:
                        f.write(str(1.0 * batch / num_batches) + "\n")
                        f.flush()

    def predict_batch(self, x: Union[PrivateTensor, SharedPair], owner):
        """

        :param x:
        :return:
        """
        with tf.name_scope("predict_batch"):
            y_hat = self.forward(x, with_sigmoid=False)

            y_hat = y_hat.to_tf_tensor(owner=owner)

            y_hat = tf.sigmoid(y_hat)
            return y_hat

    # def predict(self, sess, x, file_name, num_batches, idx, owner, record_num_ceil_mod_batch_size,
    #             y: PrivateTensor = None, new_write_file=True):
    #     """
    #
    #     :param sess:
    #     :param x:
    #     :param file_name:
    #     :param num_batches:
    #     :param idx:
    #     :param owner: output owner
    #     :param record_num_ceil_mod_batch_size:
    #     """
    #     owner = get_device(owner)
    #     predict_batch = self.predict_batch(x, owner)
    #
    #     with tf.device(owner):
    #         predict_batch = tf.strings.as_string(predict_batch)
    #         if y is None:
    #
    #             predict_batch = tf.concat([idx, predict_batch], axis=1)
    #         else:
    #             y = tf.strings.as_string(y.to_tf_tensor())
    #             predict_batch = tf.concat([idx, predict_batch, y], axis=1)
    #         predict_batch = tf.compat.v1.reduce_join(predict_batch, axis=1, separator=", ")
    #
    #         if new_write_file:
    #             open_method = "w"
    #         else:
    #             open_method = "a"
    #         if sess is not None:
    #             with open(file_name, open_method) as f:
    #                 for batch in range(num_batches):
    #                     records = sess.run(predict_batch)
    #                     if batch == num_batches - 1:
    #                         records = records[0:record_num_ceil_mod_batch_size]
    #                     records = "\n".join(records.astype('str'))
    #                     f.write(records + "\n")

    def predict(self, id, x_test, pred_batch_num, sess, predict_file=None):
        if predict_file is None:
            predict_file = StfConfig.predict_to_file
        if StfConfig.pre_produce_flag and StfConfig.offline_model:
            StfConfig.pre_produce_list = []
        y_pred = self.predict_batch(x_test, owner="R")

        if StfConfig.pre_produce_flag and StfConfig.offline_model:
            pre_produce_op = tf.group(StfConfig.pre_produce_list)
            homo_init(sess)
        random_init(sess)  # initialize the random model using in predict

        with open(predict_file, "w") as f:
            for batch in range(pred_batch_num):
                if StfConfig.pre_produce_flag and StfConfig.offline_model:
                    sess.run(pre_produce_op)
                else:
                    ids, records = sess.run([id, y_pred])
                    ids = ids.astype('str')
                    records = records.astype('str')
                    records = np.concatenate([ids, records], axis=1)
                    records = "\n".join([",".join(line) for line in records])
                    f.write(records + "\n")

    def save(self, model_file_path, model_file_machine="R"):

        if not os.path.exists(model_file_path):
            raise Exception("path not exist.{}".format(model_file_path))
        save_ops = []
        for i in range(len(self.weights)):
            model_file_path_i = os.path.join(model_file_path, "param_{i}".format(i=i))
            save_op = self.weights[i].reshape([-1]).to_private(model_file_machine).to_file(
                path=model_file_path_i, separator=", ", precision=StfConfig.to_str_precision)
            save_ops = save_ops + [save_op]
        save_op = tf.group(*save_ops)
        return save_op

    def save_as_plaintext(self, model_file_path, model_file_machine="R"):
        weights = self.b.to_private(model_file_machine).concat(self.w.reshape([-1]).to_private(model_file_machine),
                                                               axis=0)

        save_op = weights.to_file(path=model_file_path, separator=", ", precision=StfConfig.to_str_precision)
        return save_op

    def load(self, model_file_path, model_file_machine="R"):

        self.w = stf.PrivateVariable(owner=model_file_machine)
        self.b = stf.PrivateVariable(owner=model_file_machine)

        self.w.load_first_line_from_file(os.path.join(model_file_path, "param_0"), col_num=self.num_features)
        self.w = self.w.reshape([self.num_features, 1])

        self.b.load_first_line_from_file(os.path.join(model_file_path, "param_1"), col_num=1)
        self.b = self.b.reshape([1, 1])
