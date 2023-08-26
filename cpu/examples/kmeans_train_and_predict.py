#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : kmeans_train_and_predict
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-08-03 12:02
   Description : An example of MPC-secure KMeans training and prediction
   YuQue Document (Chinese version): https://yuque.antfin.com/sslab/tnoymw/srrhgb (please use Alibaba VPN)
"""

from stensorflow.ml.secure_k_means import SecureKMeans
import tensorflow as tf
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.operator.algebra import concat
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server
import time

start_local_server(config_file="../conf/config.json")
# data = np.array([[1.0, 2, 3, 4], [2, 3, 4, 6], [3, 3, 3, 3]])
np.random.seed(0)
np.set_printoptions(threshold=np.inf)

dataL = np.random.uniform(0, 10, [1280, 3])
dataR = np.random.uniform(0, 10, [1280, 3])
data = np.append(dataL, dataR, axis=1)
rows = dataL.shape[0]
StfConfig.default_fixed_point = 16
epoch = 1  # 10
k = 4

#  centers = data[np.random.choice(rows, k, replace=False)]
centers = data[0:k, :]


def tes_kmeans_secure_twoParty_iteration_batches0():
    # epoch =30
    batch_size = 128
    num_batches = len(dataL) // batch_size
    dimL = dataL.shape[1]
    dimR = dataR.shape[1]
    # k=4
    tf.compat.v1.disable_eager_execution()
    pdata_batchL = PrivateTensor("L")
    pdata_batchR = PrivateTensor("R")

    pdatasetL = tf.compat.v1.data.Dataset.from_tensor_slices(dataL)
    pdatasetL = pdatasetL.repeat(epoch * 2).batch(batch_size).make_one_shot_iterator().get_next()
    pdatasetL = tf.reshape(pdatasetL, [batch_size, dimL])
    pdatasetR = tf.compat.v1.data.Dataset.from_tensor_slices(dataR)
    pdatasetR = pdatasetR.repeat(epoch * 2).batch(batch_size).make_one_shot_iterator().get_next()
    pdatasetR = tf.reshape(pdatasetR, [batch_size, dimR])

    pdata_batchL.load_from_tf_tensor(pdatasetL)
    pdata_batchR.load_from_tf_tensor(pdatasetR)

    pdata_batch = concat([pdata_batchL, pdata_batchR], axis=1)

    centers_firstL = PrivateTensor("L")
    centers_firstR = PrivateTensor("R")

    centers_firstL.load_from_numpy(dataL[0:k, :])
    centers_firstR.load_from_numpy(dataR[0:k, :])

    centers_first = concat([centers_firstL, centers_firstR], axis=1)

    print("pdata_batch=", pdata_batch)
    print("centers_first=", centers_first)

    # sess = tf.compat.v1.Session(target=StfConfig.target)
    # sess.run(tf.compat.v1.initialize_all_variables())
    # print(sess.run(centers_first.to_tf_tensor("R")))
    # print(sess.run(pdata_batch.to_tf_tensor("R")))
    start_time = time.time()
    result4 = SecureKMeans.kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, epoch, num_batches)
    print("result4", result4)
    print("time=", time.time() - start_time)


def tes_kmeans_secure_twoParty_iteration_batches():
    from stensorflow.global_var import StfConfig
    from stensorflow.basic.basic_class.private import PrivateTensor
    from stensorflow.basic.operator.algebra import concat
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

    np.random.seed(0)  # fix the random numbers
    max_iter = 20  # 2 * k.

    # party L: initialize PrivateTensor, load data batch by batch
    pdata_first_center_L = PrivateTensor(owner='L')
    pdata_first_center_L.load_first_k_lines_from_file(path=StfConfig.train_file_onL, k=k, col_num=num_features_L
                                                      , sep=",", skip_row_num=1)

    # party R: initialize PrivateTensor, load data batch by batch
    pdata_first_center_R = PrivateTensor(owner='R')
    pdata_first_center_R.load_first_k_lines_from_file(path=StfConfig.train_file_onR, k=k, col_num=num_features_R + 1
                                                      , sep=",", skip_row_num=1)

    # split pdataR_xy to (features, label)
    pdata_first_center_R, _ = pdata_first_center_R.split(size_splits=[num_features_R, 1], axis=1)

    centers_first = concat([pdata_first_center_L, pdata_first_center_R], axis=1)
    # store the center (that is being computed) at the current epoch

    # party L: initialize PrivateTensor, load data
    pdataL = PrivateTensor(owner='L')
    pdataL.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=max_iter * 2, skip_col_num=match_col,
                          clip_value=clip_value)

    # party R: initialize PrivateTensor, load data
    pdataR_xy = PrivateTensor(owner='R')
    pdataR_xy.load_from_file(path=StfConfig.train_file_onR,
                             record_defaults=format_y, batch_size=batch_size, repeat=max_iter * 2,
                             skip_col_num=match_col,
                             clip_value=clip_value)
    # split pdataR_xy to (features, label)
    pdataRx, pdataRy = pdataR_xy.split(size_splits=[num_features_R, 1], axis=1)

    pdata_batch = concat([pdataL, pdataRx], axis=1)
    import time
    start_time = time.time()
    result4 = SecureKMeans.kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, max_iter, num_batches)
    print("end_time=", time.time() - start_time)


def tes_kmeans_secure_twoParty_iteration_batches_class():
    from stensorflow.global_var import StfConfig
    from stensorflow.basic.basic_class.private import PrivateTensor
    from stensorflow.basic.operator.algebra import concat
    from stensorflow.ml.secure_k_means import SecureKMeans

    # -------------------train -----------------------------------
    match_col = 1
    num_features_L = 5
    num_features_R = 5
    num_samples = 8429
    batch_size = 128
    clip_value = 5.0
    max_iter = 2  # 0  # 2 * k.
    format_x = [["a"]] * match_col + [[0.2]] * num_features_L
    format_y = [["a"]] * match_col + [[0.3]] * num_features_R + [[1.0]]

    num_features = num_features_L + num_features_R
    num_batches = num_samples // batch_size  # or num_samples // batch_size + 1
    np.random.seed(0)  # fix the random numbers

    # party L: initialize PrivateTensor, load data batch by batch
    pdata_first_center_L = PrivateTensor(owner='L')
    pdata_first_center_L.load_first_k_lines_from_file(path=StfConfig.train_file_onL, k=k, col_num=num_features_L
                                                      , sep=",", skip_row_num=1)

    # party R: initialize PrivateTensor, load data batch by batch
    pdata_first_center_R = PrivateTensor(owner='R')
    pdata_first_center_R.load_first_k_lines_from_file(path=StfConfig.train_file_onR, k=k, col_num=num_features_R + 1
                                                      , sep=",", skip_row_num=1)

    # split pdataR_xy to (features, label)
    pdata_first_center_R, _ = pdata_first_center_R.split(size_splits=[num_features_R, 1], axis=1)

    centers_first = concat([pdata_first_center_L, pdata_first_center_R], axis=1)

    # party L: initialize PrivateTensor, load data
    pdataL = PrivateTensor(owner='L')
    pdataL.load_from_file(path=StfConfig.train_file_onL,
                          record_defaults=format_x, batch_size=batch_size, repeat=max_iter * 2, skip_col_num=match_col,
                          clip_value=clip_value)

    # party R: initialize PrivateTensor, load data
    pdataR_xy = PrivateTensor(owner='R')
    pdataR_xy.load_from_file(path=StfConfig.train_file_onR,
                             record_defaults=format_y, batch_size=batch_size, repeat=max_iter * 2,
                             skip_col_num=match_col,
                             clip_value=clip_value)
    # split pdataR_xy to (features, label)
    pdataRx, pdataRy = pdataR_xy.split(size_splits=[num_features_R, 1], axis=1)

    pdata_batch = concat([pdataL, pdataRx], axis=1)
    start_time = time.time()
    sk = SecureKMeans(k=k, centers_first=centers_first, num_features=num_features)
    sess = tf.compat.v1.Session(target=StfConfig.target)

    sk.train(pdata_batch=pdata_batch, epoch=max_iter, batchs_in_epoch=num_batches, sess=sess)
    # result4 = sKmeans.kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, max_iter, num_batches)

    sk.save(sess, path="../output/model_kmeans")
    sk.load(path="../output/model_kmeans")

    # -----------------------predict----------------------------------------
    pdataL_test = PrivateTensor(owner='L')
    idx = pdataL_test.load_from_file_withid(path=StfConfig.pred_file_onL,
                                            record_defaults=format_x, batch_size=batch_size, repeat=max_iter * 2,
                                            id_col_num=match_col,
                                            clip_value=clip_value)

    # party R: initialize PrivateTensor, load data
    pdataR_test_xy = PrivateTensor(owner='R')
    pdataR_test_xy.load_from_file(path=StfConfig.pred_file_onR,
                                  record_defaults=format_y, batch_size=batch_size, repeat=max_iter * 2,
                                  skip_col_num=match_col,
                                  clip_value=clip_value)
    # split pdataR_xy to (features, label)
    pdataRx_test, pdataRy_test = pdataR_test_xy.split(size_splits=[num_features_R, 1], axis=1)

    pdata_batch_test = concat([pdataL_test, pdataRx_test], axis=1)

    sk.predict_to_file(sess, pdata_batch_test, StfConfig.predict_to_file, batch_num=num_batches, idx=idx)

    print("end_time=", time.time() - start_time)


if __name__ == '__main__':
    # tes_kmeans_secure_twoParty_iteration_batches0()
    # tes_kmeans_secure_twoParty_iteration_batches()
    tes_kmeans_secure_twoParty_iteration_batches_class()
