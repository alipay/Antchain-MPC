#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : spcabatch_train_and_predict
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-06-20 16:46
   Description : examples for spca_batch.py
   YuQue Document: https://yuque.antfin.com/sslab/tnoymw/vrvi78 (please use Alibaba VPN)
"""

import numpy as np
import tensorflow as tf
from stensorflow.basic.basic_class import private
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.operator.algebra import concat
from stensorflow.engine.start_server import start_local_server
from stensorflow.global_var import StfConfig
from stensorflow.ml.fully_secure_pca import FullySecurePCA
import time


def spca_twoParty_train_and_predict_v2():

    number_features_revealed = 3

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
    num_batches = num_samples // batch_size  # or num_samples // batch_size + 1
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

    pca = FullySecurePCA(batch_size=batch_size, num_features_L=num_features_L, num_features_R=num_features_R)

    sess = tf.compat.v1.Session(StfConfig.target)

    pca.train(sess, pdataL_train, pdataR_train, number_features_revealed, num_samples)

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
                                               record_defaults=format_y, batch_size=batch_size, repeat=2,
                                               id_col_num=match_col,
                                               clip_value=clip_value)

    # split pdataR_test_xy to (features, label)
    pdataR_test_x, pdataR_test_y = pdataR_test_xy.split(size_splits=[num_features_R, 1], axis=1)

    record_num_ceil_mod_batch_size = pred_record_num % batch_size
    if record_num_ceil_mod_batch_size == 0:
        record_num_ceil_mod_batch_size = batch_size
    pca.predict_to_file(sess, pdataL_test, pdataR_test_x, predict_file_name=StfConfig.predict_to_file,
                        batch_num=pred_batch_num, idx=idx, model_file_machine="R",
                        record_num_ceil_mod_batch_size=record_num_ceil_mod_batch_size)


if __name__ == '__main__':
    start_local_server(config_file="../conf/config.json")

    spca_twoParty_train_and_predict_v2()
