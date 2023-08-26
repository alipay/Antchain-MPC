#!/usr/bin/env python
# coding=utf-8

"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : spca_batch
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-08-14 15:18
   Description : fully secure principal component analysis via MPC
                 supporting two parties, defined to be L and R
                 supporting loading data batch by batch
"""

import numpy as np
import tensorflow as tf
from stensorflow.basic.operator.algebra import concat
from stensorflow.engine.start_server import start_local_server
from stensorflow.basic.basic_class import private
from stensorflow.basic.basic_class.pair import SharedVariablePair
from stensorflow.global_var import StfConfig
from stensorflow.basic.operator.inverse_sqrt import careful_inverse_sqrt
from stensorflow.basic.basic_class.pair import SharedPair


class FullySecurePCA:

    def __init__(self, batch_size, num_features_L=5, num_features_R=5):
        self.batch_size = batch_size
        self.num_features_L = num_features_L
        self.num_features_R = num_features_R
        self.number_features = self.num_features_L + self.num_features_R
        self.feature_matrix = []
        self.learning_rate = 0.1  # 0.01
        self.iteration_each_feature = 2  # 180
        self.feature_vector_normalized = None

        match_col = 1
        clip_value = 5.0
        format_x = [["a"]] * match_col + [[0.2]] * num_features_L

        # party L: initialize PrivateTensor, load data
        self.pdataL = private.PrivateTensor(owner='L')
        self.pdataL.load_from_file(path=StfConfig.train_file_onL,
                                   record_defaults=format_x, batch_size=self.batch_size, repeat=1,
                                   skip_col_num=match_col,
                                   clip_value=clip_value)

        # party R: initialize PrivateTensor and load data
        self.pdataR = private.PrivateTensor(owner='R')
        self.pdataR.load_from_file(path=StfConfig.train_file_onL,
                                   record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                                   clip_value=clip_value)

        # initialize covar_matrix_sum1 (for accumulating results of (X^T * X) batch by batch)
        self.covar_matrix_sum1 = SharedVariablePair(ownerL="L", ownerR="R",
                                                    shape=[self.number_features, self.number_features])
        self.covar_matrix_sum1.load_from_tf_tensor(tf.constant([[0] * self.number_features] * self.number_features))

        # initialize covar_matrix_sum2 (for accumulating results of (E(X))^T * E(X) batch by batch)
        self.covar_matrix_sum2 = SharedVariablePair(ownerL="L", ownerR="R",
                                                    shape=[self.number_features, self.number_features])
        self.covar_matrix_sum2.load_from_tf_tensor(tf.constant([[0] * self.number_features] * self.number_features))

        self.covar_matrix_curr = SharedVariablePair(ownerL="L", ownerR="R",
                                                    shape=[self.number_features, self.number_features])
        self.covar_matrix_curr.load_from_numpy(np.zeros([self.number_features, self.number_features]))

        self.feature_vector = SharedVariablePair(ownerL="L", ownerR="R", shape=[self.number_features, 1])
        self.feature_vector_initial = np.random.normal(size=[self.number_features, 1])
        self.feature_vector_initial = self.feature_vector_initial / np.sqrt(
            np.sum(np.power(self.feature_vector_initial, 2)))
        self.feature_vector.load_from_numpy(self.feature_vector_initial)


        # self.feature_vector_last = SharedVariablePair(ownerL="L", ownerR="R", shape=[self.number_features, 1])
        # self.feature_vector_last_initial = np.random.normal(size=[self.number_features, 1])
        # self.feature_vector_last_initial = self.feature_vector_last_initial / np.sqrt(
        #     np.sum(np.power(self.feature_vector_last_initial, 2)))
        # self.feature_vector_last.load_from_numpy(self.feature_vector_last_initial)

    def get_train_op(self, pdataL, pdataR, num_samples):
        # party L local computation -- part 1 (_p1)
        pdataL_trans = pdataL.transpose()
        covar_p1_LL = pdataL_trans @ pdataL

        # party L local computation -- part 2 (_p2)
        exp_pdataL = pdataL.reduce_sum(axis=0, keepdims=True) / num_samples
        exp_pdataL_trans = exp_pdataL.transpose()
        covar_p2_LL = exp_pdataL_trans @ exp_pdataL

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



        # construct covariance matrix
        covar_matrix = (1 / num_samples) * self.covar_matrix_sum1 - (1 / num_samples) * self.covar_matrix_sum2
        covar_matrix_curr_init_op = self.covar_matrix_curr.assign(covar_matrix)

        derivative_feature_matrix = 2 * self.covar_matrix_curr @ self.feature_vector
        feature_vector_iterative_op = self.feature_vector.assign(
            self.feature_vector + self.learning_rate * derivative_feature_matrix)

        feature_vector_square = (self.feature_vector * self.feature_vector).reduce_sum(axis=0)
        feature_vector_norm_inv = careful_inverse_sqrt(feature_vector_square, 1E-6)
        self.feature_vector_normalized = self.feature_vector * feature_vector_norm_inv

        lambda_curr = (self.feature_vector_normalized * (self.covar_matrix_curr @ self.feature_vector_normalized)).reduce_sum(
            axis=0)
        feature_transform_curr = self.feature_vector_normalized * self.feature_vector_normalized.transpose()
        covar_matrix_curr_op = self.covar_matrix_curr.assign(
            self.covar_matrix_curr - lambda_curr * feature_transform_curr)

        return [sum_ops, covar_matrix_curr_init_op, feature_vector_iterative_op, covar_matrix_curr_op]

    def train(self, sess, pdataL, pdataR, number_features_revealed, num_samples):

        num_batches = num_samples // pdataL.shape[0]

        [sum_ops,
         covar_matrix_curr_init_op,
         feature_vector_iterative_op,
         covar_matrix_curr_op] = self.get_train_op(pdataL, pdataR, num_samples)

        sess.run(tf.compat.v1.initialize_all_variables())

        sess.run(covar_matrix_curr_init_op)

        for _ in range(num_batches):
            sess.run(sum_ops)

        for i in range(number_features_revealed):  # number_features - 1
            for _ in range(self.iteration_each_feature):
                print("i,_=", i, _)
                sess.run(feature_vector_iterative_op)
                feature_vector_normalized_np = sess.run(self.feature_vector_normalized.to_tf_tensor("R"))
            self.feature_matrix.append(feature_vector_normalized_np)
            sess.run(covar_matrix_curr_op)
        self.feature_matrix = np.concatenate(self.feature_matrix, axis=1)
        print(self.feature_matrix)

    def predict(self, pdataL_test, pdataR_test):
        pdata_test = concat([pdataL_test, pdataR_test], axis=1)
        pdata_test = SharedPair.from_SharedPairBase(pdata_test)
        pca_prediction = pdata_test @ self.feature_matrix

        return pca_prediction

    def predict_to_file(self, sess, pdataL_test, pdataR_test, predict_file_name,
                        batch_num, idx, model_file_machine, record_num_ceil_mod_batch_size):
        pca_prediction = self.predict(pdataL_test, pdataR_test)
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        id_y_pred = pca_prediction.to_tf_str(owner=model_file_machine, id_col=idx)
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
        np.save(file=path + "_feature_matrix", arr=self.feature_matrix)

    def load(self, path="./output/model"):
        self.feature_matrix = np.load(file=path + '_feature_matrix.npy')
