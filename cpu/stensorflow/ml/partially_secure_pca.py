#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : partially_secure_pca
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022/8/18 下午9:22
   Description : description what the main function of this file
"""

import numpy as np
import tensorflow as tf
from stensorflow.basic.basic_class import private
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.operator.algebra import concat
from stensorflow.engine.start_server import start_local_server
from stensorflow.global_var import StfConfig


class PartiallySecurePCA:

    def __init__(self, num_features):
        self.v = None
        self.u = None
        self.s = None
        self.s_result = None
        self.u_result = None
        self.num_features = num_features
        self.covar_matrix_sum1 = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features, num_features])
        self.covar_matrix_sum1.load_from_tf_tensor(tf.constant([[0] * num_features] * num_features))

        # initialize covar_matrix_sum2 (for accumulating results of (E(X))^T * E(X) batch by batch)
        self.covar_matrix_sum2 = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features, num_features])
        self.covar_matrix_sum2.load_from_tf_tensor(tf.constant([[0] * num_features] * num_features))

    def get_train_op(self, pdataL, pdataR, num_samples):
        # # party L: initialize PrivateTensor, load data
        # pdataL = private.PrivateTensor(owner='L')
        # pdataL.load_from_file(path=StfConfig.train_file_onL,
        #                       record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
        #                       clip_value=clip_value)

        # party L local computation -- part 1 (_p1)
        pdataL_trans = pdataL.transpose()
        covar_p1_LL = pdataL_trans @ pdataL

        # party L local computation -- part 2 (_p2)
        exp_pdataL = pdataL.reduce_sum(axis=0, keepdims=True) / num_samples
        exp_pdataL_trans = exp_pdataL.transpose()
        covar_p2_LL = exp_pdataL_trans @ exp_pdataL

        # # party R: initialize PrivateTensor and load data
        # pdataR = private.PrivateTensor(owner='R')
        # pdataR.load_from_file(path=StfConfig.train_file_onL,
        #                       record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
        #                       clip_value=clip_value)

        # party R local computation -- part 1 (_p1)
        pdataR_trans = pdataR.transpose()
        covar_p1_RR = pdataR_trans @ pdataR

        # party L local computation -- part 2 (_p2)
        exp_pdataR = pdataR.reduce_sum(axis=0, keepdims=True) / num_samples
        exp_pdataR_trans = exp_pdataR.transpose()
        covar_p2_RR = exp_pdataR_trans @ exp_pdataR

        # party L and party R collaborative computation -- part 1 (_p1)
        covar_p1_LR = pdataL_trans @ pdataR
        covar_p1_RL = pdataR_trans @ pdataL

        # construct covariance matrix part 1: (X^T * X) / num_samples, X = [X_L, X_R]
        covar_matrix_p1_l1 = concat([covar_p1_LL, covar_p1_LR], axis=1)  # connect them to a row
        covar_matrix_p1_l2 = concat([covar_p1_RL, covar_p1_RR], axis=1)
        covar_matrix_p1 = concat([covar_matrix_p1_l1, covar_matrix_p1_l2], axis=0)

        # party L and party R collaborative computation -- part 2 (_p2)
        covar_p2_LR = exp_pdataL_trans @ exp_pdataR
        covar_p2_RL = exp_pdataR_trans @ exp_pdataL

        # construct covariance matrix part 2: (E(X))^T * E(X) / num_samples, X = [X_L, X_R]
        covar_matrix_p2_l1 = concat([covar_p2_LL, covar_p2_LR], axis=1)  # connect them to a row
        covar_matrix_p2_l2 = concat([covar_p2_RL, covar_p2_RR], axis=1)
        covar_matrix_p2 = concat([covar_matrix_p2_l1, covar_matrix_p2_l2], axis=0)

        # define accumulating operations
        sum_ops = [
            self.covar_matrix_sum1.assign(self.covar_matrix_sum1 + covar_matrix_p1),
            self.covar_matrix_sum2.assign(self.covar_matrix_sum2 + covar_matrix_p2)
        ]
        return sum_ops

    def train(self, sess, pdataL, pdataR, num_samples, top_k=None, expected_percentage=None):

        self.u_result = []
        self.s_result = []

        sum_ops = self.get_train_op(pdataL, pdataR, num_samples)
        num_batches = num_samples // pdataL.shape[0]
        sess.run(tf.compat.v1.initialize_all_variables())
        for i in range(num_batches):
            # print("batch ", i)
            sess.run(sum_ops)
        covar_matrix_sum1_np = sess.run(self.covar_matrix_sum1.to_tf_tensor('L'))
        covar_matrix_sum2_np = sess.run(self.covar_matrix_sum2.to_tf_tensor('L'))

        # construct covariance matrix
        covar_matrix_np = (1 / num_samples) * covar_matrix_sum1_np - (1 / num_samples) * covar_matrix_sum2_np
        [self.u, self.s, self.v] = np.linalg.svd(covar_matrix_np)

        if top_k is None and expected_percentage is None:
            print("Please set either top_k or expected_percentage.")
        elif top_k is not None and expected_percentage is None:
            if top_k >= self.num_features:
                print("Unexpected number of returned features")
            else:
                self.u = self.u[:, 0:top_k]
                # print("self.u", self.u)
                # print("self.s", self.s)
                # self.s = self.s[0:top_k]
        elif expected_percentage is not None and top_k is None:
            accumulated_entropy = 0
            overall_entropy = np.sum(self.s)
            idx = 0
            if expected_percentage >= 0.9:
                print("Unexpected percentage. Please set it less than 90%")
            else:
                while accumulated_entropy / overall_entropy < expected_percentage:
                    accumulated_entropy += self.s[idx]
                    idx += 1
            self.u = self.u[:, 0:idx]
            # self.s = self.s[0:idx]
        elif top_k is not None and expected_percentage is not None:
            print("Please set either top_k or expected_percentage.")

        # print("u", self.u)
        # print("s", self.s)

    def predict(self, pdataL_test, pdataR_test):

        # concat test data from two parties
        pdata_test = concat([pdataL_test, pdataR_test], axis=1)
        pdata_test = SharedPair.from_SharedPairBase(pdata_test)

        pcaPredict = pdata_test @ self.u
        # print("pcaPredict", pcaPredict)
        return pcaPredict

    def predict_to_file(self, sess, pdataL_test, pdataR_test, predict_file_name,
                        batch_num, idx, model_file_machine, record_num_ceil_mod_batch_size):
        pcaPredict = self.predict(pdataL_test, pdataR_test)
        # session process
        init_op = tf.compat.v1.global_variables_initializer()
        # sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
        sess.run(init_op)
        # pcaPredict_np = sess.run(pcaPredict.to_tf_tensor('R'))  # transform covar_matrix to numpy format

        id_y_pred = pcaPredict.to_tf_str(owner=model_file_machine, id_col=idx)
        sess.run(tf.compat.v1.global_variables_initializer())
        with open(predict_file_name, "w") as f:
            for batch in range(batch_num - 1):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                f.write(records + "\n")

            records = sess.run(id_y_pred)[0:record_num_ceil_mod_batch_size]
            records = "\n".join(records.astype('str'))
            f.write(records + "\n")

    def save(self, path="./output/model"):
        np.save(file=path + "_u", arr=self.u)
        np.save(file=path + "_s", arr=self.s)
        # np.save(file=path + "_v", arr=self.v)

    def load(self, path="./output/model"):
        self.u = np.load(file=path + '_u.npy')
        self.s = np.load(file=path + "_s.npy")
        # self.v = np.load(file=path + "_v.npy")
