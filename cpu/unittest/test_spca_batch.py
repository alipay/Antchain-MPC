"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_spca_batch
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-06-14 15：30
   Description : testing functions in for spca with inputting batching data
"""

import os
from stensorflow.engine.start_server import start_local_server
import time

import numpy as np
import tensorflow as tf
from stensorflow.basic.basic_class import private
from stensorflow.basic.operator.algebra import concat
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.global_var import StfConfig


def pca_compute(data):
    """
    data: a matrix, np.array
    description: plaintext computation of principal component analysis
    remark: just for testing, NOT reveal in products
    """

    # plain multiplication
    ave_data = np.mean(data, axis=0)  # get average value of each column of data
    clear_data = data - ave_data  # minus average value
    covar_matrix = np.dot(clear_data.T, clear_data)  # get covariance matrix: matrix * matrix^T

    [u, s, v] = np.linalg.svd(covar_matrix)  # Reveal covariance matrix

    return [u, s, v]


def pca_compute_twoParty(dataL, dataR):
    """
    dataL: a matrix from party L, np_array of shape [n, m_1]
    dataR: a matrix form party R, np_array of shape [n, m_2]
           [dataL, dataR]: a matrix, np_array of shape [n, m]
    description: plaintext computation of principal component analysis
        compute covariance matrix with two-party's inputs
    remark: just for testing, NOT reveal in products
    """

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
    covar_matrix = [[covar_dataLL, covar_LR], [covar_RL, covar_dataRR]]

    [u, s, v] = np.linalg.svd(covar_matrix)

    return [u, s, v]


def pca_compute_twoParty_secure(dataL, dataR):
    """
    dataL: a matrix from party L, private stf tensor of shape [n, m_1]
    dataR: a matrix form party R, private stf tensor of shape [n, m_2]
           [dataL, dataR]: a matrix, np_array of shape [n, m]
    description: plaintext computation of principal component analysis
        compute covariance matrix with two-party's inputs
        support small matrix only
    remark: just for testing, NOT reveal in products
    """

    # party L local computation
    pdataL = private.PrivateTensor(owner='L')  # initialize PrivateTensor
    pdataL.load_from_numpy(dataL)  # load dataL and transform it to a PrivateTensor
    ave_dataL = pdataL.reduce_sum(axis=0) / pdataL.shape[0]  # calculate mean
    clear_dataL = pdataL - ave_dataL
    cdataL_trans = clear_dataL.transpose()  # get transpose
    covar_dataLL = cdataL_trans @ clear_dataL  # matrix multiplication

    # party R local computation
    pdataR = private.PrivateTensor(owner='R')
    pdataR.load_from_numpy(dataR)
    ave_dataR = pdataR.reduce_sum(axis=0) / pdataR.shape[0]
    clear_dataR = pdataR - ave_dataR
    cdataR_trans = clear_dataR.transpose()
    covar_dataRR = cdataR_trans @ clear_dataR

    # party L and party R collaborative computation
    covar_LR = cdataL_trans @ clear_dataR
    covar_RL = cdataR_trans @ clear_dataL

    # construct covariance matrix
    covar_matrix_l1 = concat([covar_dataLL, covar_LR], axis=1)  # connect them to a row
    covar_matrix_l2 = concat([covar_RL, covar_dataRR], axis=1)
    covar_matrix = concat([covar_matrix_l1, covar_matrix_l2], axis=0)  # connect them to a column

    # transform PrivateTensor to TensorFlow format
    covar_matrix = covar_matrix.to_tf_tensor("R")
    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
    sess.run(init_op)
    covar_matrix_np = sess.run(covar_matrix)  # transform covar_matrix to numpy format

    [u, s, v] = np.linalg.svd(covar_matrix_np)

    return [u, s, v]


def pca_compute_twoParty_plain2(num_samples, dataL, dataR):
    """
    secure2: the second method for secure computation
    dataL: a vector/matrix from party L
    dataR: a vector/matrix form party R
           [dataL, dataR]: a matrix
    description: load data from numpy
    """


def pca_compute_twoParty_secure2(num_samples, dataL, dataR):
    """
    dataL: a matrix from party L, private stf tensor of shape [n, m_1]
    dataR: a matrix form party R, private stf tensor of shape [n, m_2]
           [dataL, dataR]: a matrix,  private stf tensor of shape [n, m]
    description: partially secure computation of principal component analysis
        compute covariance matrix with two-party's inputs
        protect covariance matrix
    remark: just for testing, NOT reveal in products
    """

    # party L local computation -- part 1 (_p1)
    pdataL = private.PrivateTensor(owner='L')
    pdataL.load_from_numpy(dataL)
    pdataL_trans = pdataL.transpose()
    covar_p1_LL = pdataL_trans @ pdataL

    # party L local computation -- part 2 (_p2)
    exp_pdataL = pdataL.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataL_trans = exp_pdataL.transpose()
    covar_p2_LL = exp_pdataL_trans @ exp_pdataL

    # party R local computation -- part 1 (_p1)
    pdataR = private.PrivateTensor(owner='R')
    pdataR.load_from_numpy(dataR)
    pdataR_trans = pdataR.transpose()
    covar_p1_RR = pdataR_trans @ pdataR

    # party R local computation -- part 2 (_p2)
    exp_pdataR = pdataR.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataR_trans = exp_pdataR.transpose()
    covar_p2_RR = exp_pdataR_trans @ exp_pdataR

    # party L and party R collaborative computation -- part 1 (_p1)
    covar_p1_LR = pdataL_trans @ pdataR
    covar_p1_RL = pdataR_trans @ pdataL

    # party L and party R collaborative computation -- part 2 (_p2)
    covar_p2_LR = exp_pdataL_trans @ exp_pdataR
    covar_p2_RL = exp_pdataR_trans @ exp_pdataL

    # construct covariance matrix part 1: (X^T * X) / num_samples, X = [X_L, X_R]
    covar_matrix_p1_l1 = concat([covar_p1_LL, covar_p1_LR], axis=1)  # connect them to a row
    covar_matrix_p1_l2 = concat([covar_p1_RL, covar_p1_RR], axis=1)
    covar_matrix_p1 = (1 / num_samples) * concat([covar_matrix_p1_l1, covar_matrix_p1_l2], axis=0)

    # construct covariance matrix part 2: (E(X))^T * E(X) / num_samples, X = [X_L, X_R]
    covar_matrix_p2_l1 = concat([covar_p2_LL, covar_p2_LR], axis=1)  # connect them to a row
    covar_matrix_p2_l2 = concat([covar_p2_RL, covar_p2_RR], axis=1)
    covar_matrix_p2 = (1 / num_samples) * concat([covar_matrix_p2_l1, covar_matrix_p2_l2], axis=0)

    # construct covariance matrix
    covar_matrix = covar_matrix_p1 - covar_matrix_p2

    # transform PrivateTensor to TensorFlow format
    covar_matrix = covar_matrix.to_tf_tensor("R")
    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
    sess.run(init_op)
    covar_matrix_np = sess.run(covar_matrix)  # transform covar_matrix to numpy format

    [u, s, v] = np.linalg.svd(covar_matrix_np)

    return [u, s, v]


def pca_compute_twoParty_secure2_batch_load_local():
    """
    dataL: a matrix from party L, private stf tensor of shape [n, m_1]
    dataR: a matrix form party R, private stf tensor of shape [n, m_2]
           [dataL, dataR]: a matrix,  private stf tensor of shape [n, m]
    description: partially secure computation of principal component analysis
        compute covariance matrix with two-party's inputs
        protect covariance matrix
        support large dataset through batch by batch
    remark: reveal in products
        plan to be online in Aug, 2022
    """

    start_local_server(config_file="../../conf/config.json")

    # parameters configuration for loading dataset: load_from_file()
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L

    num_features = num_features_L + num_features_R
    num_batches = num_samples // batch_size  # or num_samples // batch_size + 1

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
    covar_matrix_sum1 = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features, num_features])
    covar_matrix_sum1.load_from_tf_tensor(tf.constant([[0] * num_features] * num_features))

    # initialize covar_matrix_sum2 (for accumulating results of (E(X))^T * E(X) batch by batch)
    covar_matrix_sum2 = SharedVariablePair(ownerL="L", ownerR="R", shape=[num_features, num_features])
    covar_matrix_sum2.load_from_tf_tensor(tf.constant([[0] * num_features] * num_features))

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
    covar_matrix_sum1_np = sess.run(covar_matrix_sum1.to_tf_tensor('L'))
    covar_matrix_sum2_np = sess.run(covar_matrix_sum2.to_tf_tensor('L'))

    # construct covariance matrix
    covar_matrix_np = (1 / num_samples) * covar_matrix_sum1_np - (1 / num_samples) * covar_matrix_sum2_np

    [u, s, v] = np.linalg.svd(covar_matrix_np)

    return [u, s, v]


if __name__ == '__main__':
    dataL = [[1, 2], [1, 5], [1, 3]]
    dataR = [[2, 3], [2, 7], [2, 11]]

    start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

    # test 1
    result1 = pca_compute_twoParty_secure2(3, dataL, dataR)
    print(result1)

    # test 2
    time_start2 = time.time()
    result2 = pca_compute_twoParty_secure2_batch_load_local()
    time_end2 = time.time()
    time_result2 = time_end2 - time_start2
    print('result2', time_result2)

