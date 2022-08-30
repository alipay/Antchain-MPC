#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : pca_train_and_predict
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-06-20 16:46
   Description : examples for spca_batch.py
   YuQue Document (Chinese version): https://yuque.antfin.com/sslab/tnoymw/vrvi78 (please use Alibaba VPN)
"""

import numpy as np
import tensorflow as tf
from stensorflow.basic.basic_class import private
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.operator.algebra import concat
from stensorflow.engine.start_server import start_local_server
from stensorflow.global_var import StfConfig
from stensorflow.ml.partially_secure_pca import PartiallySecurePCA
import time


def spca_twoParty_train():
    """
    dataL: a matrix owned by party L, loaded from local server
    dataR: a matrix owned by party R, loaded from local server
    description: secure pca with two-party private inputs
                 support batching data
                 support loading data from local files
    """

    start_local_server(config_file="../conf/config.json")

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
    pdataL_train = private.PrivateTensor(owner='L')
    pdataL_train.load_from_file(path=StfConfig.train_file_onL,
                                record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                                clip_value=clip_value)

    # party L local computation -- part 1 (_p1)
    pdataL_trans = pdataL_train.transpose()
    covar_p1_LL = pdataL_trans @ pdataL_train

    # party L local computation -- part 2 (_p2)
    exp_pdataL = pdataL_train.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataL_trans = exp_pdataL.transpose()
    covar_p2_LL = exp_pdataL_trans @ exp_pdataL

    # party R: initialize PrivateTensor and load data
    pdataR_train = private.PrivateTensor(owner='R')
    pdataR_train.load_from_file(path=StfConfig.train_file_onL,
                                record_defaults=format_x, batch_size=batch_size, repeat=1, skip_col_num=match_col,
                                clip_value=clip_value)

    # party R local computation -- part 1 (_p1)
    pdataR_trans = pdataR_train.transpose()
    covar_p1_RR = pdataR_trans @ pdataR_train

    # party L local computation -- part 2 (_p2)
    exp_pdataR = pdataR_train.reduce_sum(axis=0, keepdims=True) / num_samples
    exp_pdataR_trans = exp_pdataR.transpose()
    covar_p2_RR = exp_pdataR_trans @ exp_pdataR

    # party L and party R collaborative computation -- part 1 (_p1)
    covar_p1_LR = pdataL_trans @ pdataR_train
    covar_p1_RL = pdataR_trans @ pdataL_train

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

    return u


def spca_twoParty_prediction():
    start_local_server(config_file="../conf/config.json")

    # parameters configuration for loading dataset: load_from_file()
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L
    format_y = [["a"]] * match_col + [[0.3]] * num_features_R + [[1.0]]

    # party L: load test data
    pdataL_test = private.PrivateTensor(owner='L')
    pdataL_test.load_from_file(path=StfConfig.pred_file_onL,
                               record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=match_col,
                               clip_value=clip_value)

    # party R: load test data
    pdataR_test_xy = private.PrivateTensor(owner='R')
    pdataR_test_xy.load_from_file(path=StfConfig.pred_file_onR,
                                  record_defaults=format_y, batch_size=batch_size, repeat=2, skip_col_num=match_col,
                                  clip_value=clip_value)

    # split pdataR_test_xy to (features, label)
    pdataR_test_x, pdataR_test_y = pdataR_test_xy.split(size_splits=[num_features_R, 1], axis=1)

    # concat test data from two parties
    pdata_test = concat([pdataL_test, pdataR_test_x], axis=1)
    pdata_test = SharedPair.from_SharedPairBase(pdata_test)

    # transform pca matrix to be private tensor
    pcaMatrix = spca_twoParty_train()
    pcaMatrixPrivate = private.PrivateTensor(owner='R')  # initialize PrivateTensor
    pcaMatrixPrivate.load_from_numpy(pcaMatrix)

    # make a prediction for test data
    time_start_pred = time.time()
    pcaPredict = pdata_test @ pcaMatrixPrivate
    time_end_pred = time.time()
    time_result3 = time_end_pred - time_start_pred
    print('pred time', time_result3)

    # session process
    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session("grpc://0.0.0.0:8887")
    sess.run(init_op)
    pcaPredict_np = sess.run(pcaPredict.to_tf_tensor('R'))  # transform covar_matrix to numpy format

    return pcaPredict_np


def spca_twoParty_train_and_predict():
    start_local_server(config_file="../conf/config.json")

    # parameters configuration for loading dataset: load_from_file()
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L
    format_y = [["a"]] * match_col + [[0.3]] * num_features_R + [[1.0]]

    num_features = num_features_L + num_features_R
    # num_batches = num_samples // batch_size  # or num_samples // batch_size + 1
    pred_record_num = 12042 * 3 // 10
    pred_batch_num = pred_record_num // batch_size + 1

    # party L: initialize PrivateTensor, load data
    pdataL_train = private.PrivateTensor(owner='L')
    pdataL_train.load_from_file(path=StfConfig.train_file_onL,
                                record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=match_col,
                                clip_value=clip_value)

    # party R: initialize PrivateTensor and load data
    pdataR_train = private.PrivateTensor(owner='R')
    pdataR_train.load_from_file(path=StfConfig.train_file_onL,
                                record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=match_col,
                                clip_value=clip_value)

    pca = PartiallySecurePCA(num_features=num_features)

    sess = tf.compat.v1.Session(StfConfig.target)

    top_k = 3
    expected_percentage = None

    pca.train(sess, pdataL_train, pdataR_train, num_samples, top_k, expected_percentage)

    pca.save(path="../output/model")
    pca.load(path="../output/model")
    # -------------------predict --------------------------
    pdataL_test = private.PrivateTensor(owner='L')
    pdataL_test.load_from_file(path=StfConfig.pred_file_onL,
                               record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=match_col,
                               clip_value=clip_value)

    # party R: load test data
    pdataR_test_xy = private.PrivateTensor(owner='R')
    idx = pdataR_test_xy.load_from_file_withid(path=StfConfig.pred_file_onR,
                                  record_defaults=format_y, batch_size=batch_size, repeat=2, id_col_num=match_col,
                                  clip_value=clip_value)

    # split pdataR_test_xy to (features, label)
    pdataR_test_x, pdataR_test_y = pdataR_test_xy.split(size_splits=[num_features_R, 1], axis=1)

    record_num_ceil_mod_batch_size = pred_record_num % batch_size
    if record_num_ceil_mod_batch_size == 0:
        record_num_ceil_mod_batch_size = batch_size
    pca.predict_to_file(sess, pdataL_test, pdataR_test_x, predict_file_name=StfConfig.predict_to_file,
                        batch_num=pred_batch_num, idx=idx, model_file_machine="R",
                        record_num_ceil_mod_batch_size=record_num_ceil_mod_batch_size)

    # print("record_num_ceil_mod_batch_size", record_num_ceil_mod_batch_size)


if __name__ == '__main__':
    spca_twoParty_train_and_predict()
