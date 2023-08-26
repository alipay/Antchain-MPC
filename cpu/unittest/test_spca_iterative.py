#!/usr/bin/env python
# coding=utf-8

"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_spca_iterative
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-06-14 15:18
   Description : secure principal component analysis via MPC
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


def pca_basic_backward(covar_matrix, learning_rate, feature_vector_last):
    """
    covar_matrix:
    learning_rate:
    feature_matrix_last：
    description: backward propagation of plaintext PCA
    """

    sum_of_eigenvalues = np.trace(covar_matrix, offset=0)
    print("sum_of_eigenvalues", sum_of_eigenvalues)
    vector_norm_bias = np.linalg.norm(feature_vector_last, ord=2) - 1
    print("vector_norm_bias", vector_norm_bias)
    if vector_norm_bias < 0:
        relu_bias = 0
        return feature_vector_last, 0
    elif vector_norm_bias >= 0:
        relu_bias = 1
    # relu_bias = tf.nn.relu(matrix_norm_bias)
    derivative_feature_matrix = 2 * np.dot(covar_matrix,
                                           feature_vector_last) - 2 * sum_of_eigenvalues * relu_bias * feature_vector_last
    print("derivative_feature_matrix", derivative_feature_matrix)
    new_feature_vector = feature_vector_last + learning_rate * derivative_feature_matrix
    print("new_feature_matrix", new_feature_vector)

    return new_feature_vector, 1


def pca_basic_backward1(covar_matrix, learning_rate, feature_vector_last):
    """
    covar_matrix:
    learning_rate:
    feature_matrix_last：
    description: backward propagation of plaintext PCA
    """

    derivative_feature_matrix = 2 * np.dot(covar_matrix,
                                           feature_vector_last)
    # - 2 * sum_of_eigenvalues * relu_bias * feature_vector_last
    new_feature_vector = feature_vector_last + learning_rate * derivative_feature_matrix
    new_feature_vector = new_feature_vector / np.linalg.norm(new_feature_vector, ord=2)
    return new_feature_vector


def pca_basic_iterative(dataL, dataR):
    """
    dataL: a matrix from party L, nparray of shape [num_samples, m_1]
    dataR: a matrix from party L, nparray of shape [num_samples, m_2]
           data = [dataL, dataR], num_features = m_1 + m_2
    description: solve plaintext PCA using iterative solution
    """
    learning_rate = 0.1
    number_features = 6
    iteration_each_feature = 10

    # initialize the first one using random matrix X s.t. |X|=1
    # feature_matrix = np.random.uniform(0, 1, [6, 6])

    feature_matrix = np.empty([number_features, number_features])

    # party L local computation
    ave_dataL = np.mean(dataL, axis=0)
    clear_dataL = dataL - ave_dataL
    cdataL_trans = clear_dataL.T
    covar_dataLL = np.dot(cdataL_trans, clear_dataL)

    # party R local computation
    ave_dataR = np.mean(dataR, axis=0)
    clear_dataR = dataR - ave_dataR
    cdataR_trans = clear_dataR.T
    covar_dataRR = np.dot(cdataR_trans, clear_dataR)

    # party L and party R collaborative computation
    covar_LR = np.dot(cdataL_trans, clear_dataR)
    covar_RL = np.dot(cdataR_trans, clear_dataL)

    # construct covariance matrix
    covar_matrix_1 = np.concatenate((covar_dataLL, covar_LR), axis=1)
    covar_matrix_2 = np.concatenate((covar_RL, covar_dataRR), axis=1)
    covar_matrix_curr = np.concatenate((covar_matrix_1, covar_matrix_2), axis=0)
    covar_matrix_init = covar_matrix_curr

    for i in range(1):
        feature_vector_0 = np.zeros(number_features - 1)
        feature_vector = np.append(feature_vector_0, 1)
        feature_vector = np.random.normal(size=[number_features])
        feature_vector = feature_vector / np.sqrt(np.sum(np.power(feature_vector,2 )))
        # np.linalg.norm(feature_vector,)
        for _ in range(iteration_each_feature):
            # feature_vector, relu_bias = pca_basic_backward(covar_matrix_curr, learning_rate, feature_vector)
            feature_vector = pca_basic_backward1(covar_matrix_curr, learning_rate, feature_vector)

        feature_matrix[i] = feature_vector
        print("feature_vector", feature_vector)
        print("A=", covar_matrix_curr)
        print("Ax=", covar_matrix_curr @ feature_vector)
        print("Ax/x=", (covar_matrix_curr @ feature_vector) / feature_vector)
        lambda_curr = np.sum(feature_vector * (covar_matrix_curr @ feature_vector))
        covar_matrix_curr = covar_matrix_curr - lambda_curr * feature_vector * feature_vector.transpose()

        print("lambda_curr", lambda_curr)
        print("covar_matrix_curr", covar_matrix_curr)

    [u, s, v] = np.linalg.svd(covar_matrix_init)  # for comparison

    return feature_matrix


def pca_basic_iterative(dataL, dataR):
    """
    dataL: a matrix from party L, nparray of shape [num_samples, m_1]
    dataR: a matrix from party L, nparray of shape [num_samples, m_2]
           data = [dataL, dataR], num_features = m_1 + m_2
    description: solve plaintext PCA using iterative solution
    """
    learning_rate = 0.01
    number_features = 4
    iteration_each_feature = 180

    # initialize the first one using random matrix X s.t. |X|=1
    # feature_matrix = np.random.uniform(0, 1, [6, 6])

    feature_matrix = np.empty([number_features, number_features])

    # party L local computation
    ave_dataL = np.mean(dataL, axis=0)
    clear_dataL = dataL - ave_dataL
    cdataL_trans = clear_dataL.T
    covar_dataLL = np.dot(cdataL_trans, clear_dataL)

    # party R local computation
    ave_dataR = np.mean(dataR, axis=0)
    clear_dataR = dataR - ave_dataR
    cdataR_trans = clear_dataR.T
    covar_dataRR = np.dot(cdataR_trans, clear_dataR)

    # party L and party R collaborative computation
    covar_LR = np.dot(cdataL_trans, clear_dataR)
    covar_RL = np.dot(cdataR_trans, clear_dataL)

    # construct covariance matrix
    covar_matrix_1 = np.concatenate((covar_dataLL, covar_LR), axis=1)
    covar_matrix_2 = np.concatenate((covar_RL, covar_dataRR), axis=1)
    covar_matrix_curr = np.concatenate((covar_matrix_1, covar_matrix_2), axis=0)
    covar_matrix_curr_initial = covar_matrix_curr

    [u, s, v] = np.linalg.svd(covar_matrix_curr_initial)
    print("ground_truth", v)

    # our iterative solution
    for i in range(number_features - 1):
        feature_vector = np.random.normal(size=[number_features])
        feature_vector = feature_vector / np.sqrt(np.sum(np.power(feature_vector, 2)))

        for _ in range(iteration_each_feature):
            derivative_feature_matrix = 2 * np.dot(covar_matrix_curr, feature_vector)
            feature_vector = feature_vector + learning_rate * derivative_feature_matrix

            feature_vector = feature_vector / np.linalg.norm(feature_vector, ord=2)

        feature_matrix[i].load_from_SharedPair(feature_vector)
        lambda_curr = np.sum(feature_vector * (covar_matrix_curr @ feature_vector))
        feature_matrix_curr = np.expand_dims(feature_vector, axis=1) * np.expand_dims(feature_vector, axis=0)
        covar_matrix_curr = covar_matrix_curr - lambda_curr * feature_matrix_curr

        if True:
            feature_vector = np.random.normal(size=[number_features])
            feature_vector = feature_vector / np.sqrt(np.sum(np.power(feature_vector, 2)))
            for _ in range(iteration_each_feature):
                derivative_feature_matrix = 2 * np.dot(covar_matrix_curr_initial, feature_vector)
                feature_vector = feature_vector - learning_rate * derivative_feature_matrix

                feature_vector = feature_vector / np.linalg.norm(feature_vector, ord=2)
            feature_matrix[number_features - 1] = feature_vector

    return feature_matrix


def pca_MPC_secure_iterative_prepare():
    start_local_server(config_file="../conf/config.json")

    # parameters configuration for loading dataset: load_from_file()
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    number_features = num_features_L + num_features_R
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L
    num_batches = num_samples // batch_size  # or num_samples // batch_size + 1

    iteration_each_feature = 180
    learning_rate = 0.01

    feature_matrix = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    feature_matrix.load_from_numpy(np.empty([number_features, 1], dtype=int))
    # feature_matrix = np.empty([number_features, number_features])

    # party L: initialize PrivateTensor, load data
    pdataL = private.PrivateTensor(owner='L')
    pdataL.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                          clip_value=clip_value)

    # party L local computation -- part 1 (_p1)
    pdataL_trans = pdataL.transpose()
    covar_p1_LL = pdataL_trans @ pdataL

    # party L local computation -- part 2 (_p2)
    exp_pdataL = pdataL.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataL_trans = exp_pdataL.transpose()
    covar_p2_LL = exp_pdataL_trans @ exp_pdataL

    # party R: initialize PrivateTensor and load data
    pdataR = private.PrivateTensor(owner='R')
    pdataR.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                          clip_value=clip_value)

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

    # initialize covar_matrix_sum1 (for accumulating results of (X^T * X) batch by batch)
    covar_matrix_sum1 = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    covar_matrix_sum1.load_from_tf_tensor(tf.constant([[0] * number_features] * number_features))

    # initialize covar_matrix_sum2 (for accumulating results of (E(X))^T * E(X) batch by batch)
    covar_matrix_sum2 = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    covar_matrix_sum2.load_from_tf_tensor(tf.constant([[0] * number_features] * number_features))

    # define accumulating operations
    sum_ops = [
        covar_matrix_sum1.assign(covar_matrix_sum1 + covar_matrix_p1),
        covar_matrix_sum2.assign(covar_matrix_sum2 + covar_matrix_p2)
    ]

    # initialize session
    sess = tf.compat.v1.Session(StfConfig.target)
    init_op = tf.compat.v1.initialize_all_variables()
    sess.run(init_op)

    # perform accumulation
    for _ in range(num_batches):
        sess.run(sum_ops)

    # construct covariance matrix
    covar_matrix_curr = (1 / num_samples) * covar_matrix_sum1 - (1 / num_samples) * covar_matrix_sum2

    covar_matrix_curr_initial = covar_matrix_curr

    for i in range(3):  # number_features - 1
        feature_vector = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, 1])
        feature_vector_initial = np.random.normal(size=[number_features])
        feature_vector_initial = feature_vector_initial / np.sqrt(np.sum(np.power(feature_vector_initial, 2)))
        feature_vector.load_from_numpy(feature_vector_initial)

        feature_vector_normalized = None
        feature_vector = feature_vector.expend_dims(axis=1)

        for _ in range(iteration_each_feature):
            derivative_feature_matrix = 2 * covar_matrix_curr @ feature_vector
            feature_vector = feature_vector + learning_rate * derivative_feature_matrix

            feature_vector_square = (feature_vector * feature_vector).reduce_sum(axis=0)
            feature_vector_norm_inv = careful_inverse_sqrt(feature_vector_square, 1E-6)
            feature_vector_normalized = feature_vector * feature_vector_norm_inv

        feature_matrix = concat([feature_matrix, feature_vector_normalized], axis=1)
        lambda_curr = (feature_vector_normalized * (covar_matrix_curr @ feature_vector_normalized)).reduce_sum(axis=0)
        feature_matrix_curr = feature_vector_normalized * feature_vector_normalized.transpose()
        covar_matrix_curr = covar_matrix_curr - lambda_curr * feature_matrix_curr

    if True:
        feature_vector_last_normalized = None
        feature_vector_last_initial = np.random.normal(size=[number_features])
        feature_vector_last_initial = feature_vector_last_initial / np.sqrt(
            np.sum(np.power(feature_vector_last_initial, 2)))

        feature_vector_last = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, 1])
        feature_vector_last.load_from_numpy(feature_vector_last_initial)
        feature_vector_last = feature_vector_last.expend_dims(axis=1)

        for _ in range(iteration_each_feature):
            derivative_feature_matrix = 2 * covar_matrix_curr_initial @ feature_vector_last
            feature_vector_last = feature_vector_last - learning_rate * derivative_feature_matrix

            feature_vector_last_square = (feature_vector_last * feature_vector_last).reduce_sum(axis=0)
            feature_vector_last_norm_inv = careful_inverse_sqrt(feature_vector_last_square, 1E-6)
            feature_vector_last_normalized = feature_vector_last * feature_vector_last_norm_inv

        feature_matrix = concat([feature_matrix, feature_vector_last_normalized], axis=1)

    return feature_matrix


def pca_MPC_secure_iterative():
    start_local_server(config_file="../conf/config.json")

    # parameters configuration for loading dataset: load_from_file()
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    number_features = num_features_L + num_features_R
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L
    num_batches = num_samples // batch_size  # or num_samples // batch_size + 1

    iteration_each_feature = 6  # 180
    learning_rate = 0.01

    number_features_revealed = 9

    feature_matrix = []
    # feature_matrix = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    # feature_matrix.load_from_numpy(np.empty([number_features, 1], dtype=int))
    # feature_matrix = np.empty([number_features, number_features])

    # party L: initialize PrivateTensor, load data
    pdataL = private.PrivateTensor(owner='L')
    pdataL.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                          clip_value=clip_value)

    # party L local computation -- part 1 (_p1)
    pdataL_trans = pdataL.transpose()
    covar_p1_LL = pdataL_trans @ pdataL

    # party L local computation -- part 2 (_p2)
    exp_pdataL = pdataL.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataL_trans = exp_pdataL.transpose()
    covar_p2_LL = exp_pdataL_trans @ exp_pdataL

    # party R: initialize PrivateTensor and load data
    pdataR = private.PrivateTensor(owner='R')
    pdataR.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                          clip_value=clip_value)

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

    # initialize covar_matrix_sum1 (for accumulating results of (X^T * X) batch by batch)
    covar_matrix_sum1 = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    covar_matrix_sum1.load_from_tf_tensor(tf.constant([[0] * number_features] * number_features))

    # initialize covar_matrix_sum2 (for accumulating results of (E(X))^T * E(X) batch by batch)
    covar_matrix_sum2 = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    covar_matrix_sum2.load_from_tf_tensor(tf.constant([[0] * number_features] * number_features))

    # define accumulating operations
    sum_ops = [
        covar_matrix_sum1.assign(covar_matrix_sum1 + covar_matrix_p1),
        covar_matrix_sum2.assign(covar_matrix_sum2 + covar_matrix_p2)
    ]

    covar_matrix_curr = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, number_features])
    covar_matrix_curr.load_from_numpy(np.zeros([number_features, number_features]))

    feature_vector = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, 1])
    feature_vector_initial = np.random.normal(size=[number_features, 1])
    feature_vector_initial = feature_vector_initial / np.sqrt(np.sum(np.power(feature_vector_initial, 2)))
    feature_vector.load_from_numpy(feature_vector_initial)

    feature_vector_last = SharedVariablePair(ownerL="L", ownerR="R", shape=[number_features, 1])
    feature_vector_last_initial = np.random.normal(size=[number_features, 1])
    feature_vector_last_initial = feature_vector_last_initial / np.sqrt(np.sum(np.power(feature_vector_last_initial, 2)))
    feature_vector_last.load_from_numpy(feature_vector_last_initial)

    # construct covariance matrix
    covar_matrix = (1 / num_samples) * covar_matrix_sum1 - (1 / num_samples) * covar_matrix_sum2
    #covar_matrix_curr.load_from_SharedPair(covar_matrix)
    covar_matrix_curr_init_op = covar_matrix_curr.assign(covar_matrix)

    # feature_vector_initialized_op = feature_vector.assign(feature_vector_initial)

    derivative_feature_matrix = 2 * covar_matrix_curr @ feature_vector
    feature_vector_iterative_op = feature_vector.assign(feature_vector + learning_rate * derivative_feature_matrix)

    feature_vector_square = (feature_vector * feature_vector).reduce_sum(axis=0)
    feature_vector_norm_inv = careful_inverse_sqrt(feature_vector_square, 1E-6)
    feature_vector_normalized = feature_vector * feature_vector_norm_inv

    lambda_curr = (feature_vector_normalized * (covar_matrix_curr @ feature_vector_normalized)).reduce_sum(axis=0)
    feature_transform_curr = feature_vector_normalized * feature_vector_normalized.transpose()
    covar_matrix_curr_op = covar_matrix_curr.assign(covar_matrix_curr - lambda_curr * feature_transform_curr)

    # derivative_feature_matrix = 2 * covar_matrix_curr_initial @ feature_vector_last
    # feature_vector_last_op = feature_vector_last.assign(feature_vector_last - learning_rate * derivative_feature_matrix)

    # feature_vector_last_square = (feature_vector_last * feature_vector_last).reduce_sum(axis=0)
    # feature_vector_last_norm_inv = careful_inverse_sqrt(feature_vector_last_square, 1E-6)
    # feature_vector_last_normalized = feature_vector_last * feature_vector_last_norm_inv

    # initialize session
    sess = tf.compat.v1.Session(StfConfig.target)
    init_op = tf.compat.v1.initialize_all_variables()
    sess.run(init_op)
    sess.run(covar_matrix_curr_init_op)
    for _ in range(num_batches):
        print("l369, _=", _)
        sess.run(sum_ops)

    for i in range(number_features_revealed):  # number_features - 1
        for _ in range(iteration_each_feature):
            print("i,_=", i, _)
            sess.run(feature_vector_iterative_op)
            feature_vector_normalized_np = sess.run(feature_vector_normalized.to_tf_tensor("R"))
        feature_matrix.append(feature_vector_normalized_np)
        sess.run(covar_matrix_curr_op)
    print(feature_matrix)
    return feature_matrix


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    dataL = [[1, 2], [1, 7], [2, 5], [9, 5]]
    dataR = [[2, 7], [3, 9], [3, 3], [4, 4]]

    result3 = pca_MPC_secure_iterative()
    print("result3", result3)

