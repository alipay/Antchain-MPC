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
import time
from stensorflow.basic.basic_class.pair import SharedPair, SharedVariablePair
from stensorflow.basic.basic_class.base import get_device
from stensorflow.basic.basic_class.private import PrivateTensor
import numpy as np
import os
from stensorflow.basic.operator.sigmoid import sigmoid_sin
from stensorflow.global_var import StfConfig
from stensorflow.homo_enc.homo_enc import homo_init
from stensorflow.random.random import random_init

class LogisticRegression2:
    """Contains methods to build and train logistic regression."""

    def __init__(self, num_features_L, num_features_R, learning_rate=0.01, l2_regularzation=1E-2):
        """
        :param num_features_L:  excluding label. part X -> L
        :param num_features_R:  including label. part Y -> R
        :param learning_rate:
        :param l2_regularzation:
        """
        # self.w = tfe.define_private_variable(
        #     tf.random_uniform([num_features, 1], -0.01, 0.01))
        assert (num_features_L >= 0 and num_features_R >= 0), "features should be non-negative"
        self.num_features_L = num_features_L
        self.num_features_R = num_features_R
        self.num_features = num_features_L + num_features_R
        # self.w = tfe.define_private_variable(
        #     tf.random_uniform([num_features, 1], -1.0/np.sqrt(num_features), 1.0/np.sqrt(num_features)))
        self.w_L = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features_L, 1])
        self.w_R = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features_R, 1])
        # self.w = SharedVariablePair(ownerL="L", ownerR="R")
        # self.w_L.load_from_tf_tensor(tf.compat.v1.random_uniform([num_features_L, 1],
        #                                                     -1.0/np.sqrt(num_features_L + 1),
        #                                                     1.0/np.sqrt(num_features_L + 1)))
        # self.w_R.load_from_tf_tensor(tf.compat.v1.random_uniform([num_features_R, 1],
        #                                                     -1.0 / np.sqrt(num_features_R + 1),
        #                                                     1.0 / np.sqrt(num_features_R + 1)))
        self.w_L.load_from_numpy(np.random.uniform(size=[num_features_L, 1],
                                                   low=-1.0 / np.sqrt(num_features_L + 1),
                                                   high=1.0 / np.sqrt(num_features_L + 1)))
        self.w_R.load_from_numpy(np.random.uniform(size=[num_features_R, 1],
                                                   low=-1.0 / np.sqrt(num_features_R + 1),
                                                   high=1.0 / np.sqrt(num_features_R + 1)))

        self.b = SharedVariablePair(ownerL="L", ownerR="R", shape=[1])
        # self.b.load_from_tf_tensor(tf.zeros([1]))
        self.b.load_from_numpy(np.zeros(shape=[1]))
        self.learning_rate = learning_rate
        self.l2_regularzation = l2_regularzation

    @property
    def weights(self):
        """

        :return:
        """
        return self.w_L, self.w_R, self.b

    def forward(self,
                x_L: Union[PrivateTensor, SharedPair],
                x_R: Union[PrivateTensor, SharedPair],
                with_sigmoid=True):
        """

        :param x:
        :param with_sigmoid:
        :return:
        """

        m = x_L @ self.w_L + x_R @ self.w_R
        out = m + self.b

        if with_sigmoid:
            y = sigmoid_sin(out, M=16)
        else:
            y = out
        return y

    def backward(self,
                 x_L: Union[PrivateTensor, SharedPair],
                 x_R: Union[PrivateTensor, SharedPair],
                 dy: Union[PrivateTensor, SharedPair], learning_rate=0.01):
        """
        :param x_L:
        :param x_R:
        :param dy:
        :param learning_rate:
        :return:
        """

        batch_size = x_L.shape[0]
        with tf.name_scope("backward"):
            dw_L = (x_L.transpose() @ dy) / batch_size + self.l2_regularzation * self.w_L
            dw_R = (x_R.transpose() @ dy) / batch_size + self.l2_regularzation * self.w_R
            db = dy.reduce_sum(axis=0) / batch_size
            assign_ops = [
                self.w_L.assign(self.w_L - learning_rate * dw_L),
                self.w_R.assign(self.w_R - learning_rate * dw_R),
                self.b.assign(self.b - learning_rate * db),
            ]
            return assign_ops

    def loss_grad(self, y: Union[PrivateTensor, SharedPair], y_hat: Union[PrivateTensor, SharedPair]):
        """

        :param y:
        :param y_hat:
        :return:
        """
        with tf.name_scope("loss-grad"):
            dy = y_hat - y
            return dy

    def fit_batch(self,
                  x_L: Union[PrivateTensor, SharedPair],
                  x_R: Union[PrivateTensor, SharedPair],
                  y: Union[PrivateTensor, SharedPair]):
        """

        :param x:
        :param y:
        :return:
        """
        with tf.name_scope("fit-batch"):
            y_hat = self.forward(x_L=x_L, x_R=x_R, with_sigmoid=True)
            dy = self.loss_grad(y, y_hat)
            fit_batch_op = self.backward(x_L, x_R, dy, self.learning_rate)
            return fit_batch_op

    def fit(self, sess,
            x_L: Union[PrivateTensor, SharedPair],
            x_R: Union[PrivateTensor, SharedPair],
            y: Union[PrivateTensor, SharedPair],
            num_batches):
        """

        :param sess:
        :param x_L: x part features
        :param x_R: y part features
        :param y:
        :param num_batches:
        :param progress_file:
        """
        fit_batch_op = self.fit_batch(x_L, x_R, y)
        if StfConfig.pre_produce_flag and StfConfig.offline_model:
            pre_produce_op = tf.group(StfConfig.pre_produce_list)
        if sess is not None:
            sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        if sess is not None:
            for batch in range(num_batches):
                print("batch", batch)
                if StfConfig.pre_produce_flag and StfConfig.offline_model:
                    sess.run(pre_produce_op)
                else:
                    sess.run(fit_batch_op)
        end_time = time.time()
        print("train time=", end_time - start_time)

    def predict_batch(self,
                      x_L: Union[PrivateTensor, SharedPair],
                      x_R: Union[PrivateTensor, SharedPair],
                      owner):
        """
        :param x_L:
        :param x_R:
        :return:
        """
        y_hat = self.forward(x_L=x_L, x_R=x_R, with_sigmoid=False)

        y_hat = y_hat.to_tf_tensor(owner=owner)

        # y_hat = y_hat / tf.cast(x.shape[1], dtype='float32')
        y_hat = tf.sigmoid(y_hat)
        return y_hat

    # def predict(self, sess, x_L, x_R, file_name, num_batches, idx,
    #             progress_file, owner, record_num_ceil_mod_batch_size):
    #     """
    #
    #     :param sess:
    #     :param x_L:
    #     :param x_R:
    #     :param file_name:  write prediction
    #     :param num_batches:
    #     :param idx:
    #     :param progress_file:
    #     :param owner: output owner
    #     :param record_num_ceil_mod_batch_size:
    #     """
    #
    #     owner = get_device(owner)
    #     predict_batch = self.predict_batch(x_L=x_L, x_R=x_R, owner=owner)
    #
    #     with tf.device(owner):
    #         predict_batch = tf.strings.as_string(predict_batch)
    #
    #         predict_batch = tf.concat([idx, predict_batch], axis=1)
    #         predict_batch = tf.compat.v1.reduce_join(predict_batch, axis=1, separator=", ")
    #         # predict_batch=tf.reduce_join(predict_batch, separator="\n")
    #         if sess is not None:
    #             with open(file_name, "w") as f, open(progress_file, "a") as progress_file:
    #                 for batch in range(num_batches):
    #
    #                     records = sess.run(predict_batch)
    #
    #                     if batch == num_batches - 1:
    #                         records = records[0:record_num_ceil_mod_batch_size]
    #
    #                     # records = str(records, encoding="utf8")
    #                     records = "\n".join(records.astype('str'))
    #                     # records.to_file()
    #                     f.write(records + "\n")
    #
    #                     # if (batch % 10 == 0):
    #                     if batch % (1 + int(num_batches / 100)) == 0:
    #                         progress_file.write(str(1.0 * batch / num_batches) + "\n")
    #                         progress_file.flush()

    def predict(self, id, xL_test, xR_test, pred_batch_num, sess, predict_file=None):
        if predict_file is None:
            predict_file = StfConfig.predict_to_file
        if StfConfig.pre_produce_flag and StfConfig.offline_model:
            StfConfig.pre_produce_list = []
        y_pred = self.predict_batch(x_L=xL_test, x_R=xR_test, owner="R")
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
            # save_op = tfe.define_output(modelFileMachine, [self.weights[i], modelFilePath_i], _save)
            save_op = self.weights[i].reshape([-1]).to_private(model_file_machine).to_file(
                path=model_file_path_i, separator=", ", precision=StfConfig.to_str_precision)
            save_ops = save_ops + [save_op]
        save_op = tf.group(*save_ops)
        return save_op

    def save_as_plaintext(self, model_file_path, model_file_machine="R"):
        private_b = self.b.to_private(model_file_machine)
        private_w_L = self.w_L.reshape([-1]).to_private(model_file_machine)
        private_w_R = self.w_R.reshape([-1]).to_private(model_file_machine)
        # CommonStf_config.default_logger.info("model_file_path:{}, shape, private_b:{}, w_L:{}, w_R:{}".format(
        #     model_file_path, private_b.shape, private_w_L.shape, private_w_R.shape))
        weights = private_b.concat(private_w_L, axis=0).concat(private_w_R, axis=0)
        save_op = weights.to_file(path=model_file_path, separator=", ", precision=StfConfig.to_str_precision)
        return save_op

    def load(self, model_file_path, model_file_machine="R"):
        self.w_L = stf.PrivateVariable(owner=model_file_machine)
        self.w_R = stf.PrivateVariable(owner=model_file_machine)
        self.b = stf.PrivateVariable(owner=model_file_machine)

        self.w_L.load_first_line_from_file(os.path.join(model_file_path, "param_0"), col_num=self.num_features_L)
        self.w_L = self.w_L.reshape([self.num_features_L, 1])

        self.w_R.load_first_line_from_file(os.path.join(model_file_path, "param_1"), col_num=self.num_features_R)
        self.w_R = self.w_R.reshape([self.num_features_R, 1])

        self.b.load_first_line_from_file(os.path.join(model_file_path, "param_2"), col_num=1)
        self.b = self.b.reshape([1, 1])
