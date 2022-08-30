#!/usr/bin/env python
# coding=utf-8

"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_sKmeans
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-08-03 12:02
   Description : testing functions in file /stensorflow/ml/sKmeans.py
"""

from stensorflow.basic.basic_class.private import PrivateTensor
import numpy as np
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
from stensorflow.global_var import StfConfig
from stensorflow.basic.operator.algebra import concat
from stensorflow.basic.operator.argmax import argmin
from stensorflow.basic.basic_class.pair import SharedPair, SharedVariablePair
from sklearn.cluster import KMeans


def kmeans_sklearn(data, centers_init):
    """
    data: {array-like, sparse matrix} of shape
    centers_init : ndarray of shape
    Description: plaintext KMeans using sklearn
    Remark: just for testing; NOT reveal in business product
    """

    kmeans_object = KMeans(n_clusters=3, random_state=0, init=centers_init)
    kmeans_object.fit(data)
    print(kmeans_object.labels_)
    print(kmeans_object.predict([[2, 2, 2], [100, 50, 44]]))


def kmeans_basic(data, k, centers, epoch):
    """
    k: the number of centers, int
    data: {array-like, sparse matrix} of shape m x n
          m is the number of samples
          n is the number of attributes/features
    Description: plaintext KMeans using self-implementation
    Remark: just for testing; NOT reveal in business product
    """

    rows, columns = np.shape(data)
    result = np.empty(rows, dtype=np.int)
    # centers = data[np.random.choice(np.arange(rows), k, replace=False)]  # randomly chose k rows to initialize centers

    while True:  # for _ in range(epoch):

        # compute distance to every center for each sample
        dist = np.square(np.repeat(data, k, axis=0).reshape(rows, k, columns) - centers)
        # distance = np.sqrt(np.sum(dist, axis=2))  # sqrt distance to centers for each sample
        distance = np.sum(dist, axis=2)

        # find min(distance) to know which center the sample belongs to
        index_min = np.argmin(distance, axis=1)

        # optimum: if centers are the final results (this condition is very strong)
        if (index_min == result).all():
            return result, centers

        # compute the new centers for next epoch
        result[:] = index_min
        for i in range(k):
            items = data[result == i]
            centers[i] = np.mean(items, axis=0)

    # return result, centers  # if using epoch not while true


def kmeans_basic_twoParty(dataL, dataR, k, centers):
    """
    k: the number of centers, int
    data: {array-like, sparse matrix} of shape m x n
        m is the number of samples
        n is the number of attributes/features
    Description: plaintext KMeans using self-implementation
        assign data to two parties
    Remark: just for testing; NOT reveal in business product
    """

    # initialization and parameter configuration
    data = np.append(dataL, dataR, axis=1)
    rows, columns = np.shape(data)
    result = np.empty(rows, dtype=np.int)
    max_iter = k  # the max number of iterations

    # if no centers, initialize it using the following code
    # size_centers = k * columns
    # centers = data[np.random.choice(np.arange(rows), k, replace=False)]
    # centers_counter = np.empty(size_centers, dtype=np.int).reshape(k, columns)

    for _ in range(max_iter):
        # compute distance to a center for each sample
        dist = np.square(np.repeat(data, k, axis=0).reshape(rows, k, columns) - centers)
        distance = np.sum(dist, axis=2)

        # find min(distance) and identify which center the samples belong to
        index_min = np.argmin(distance, axis=1)
        result[:] = index_min

        for j in range(k):
            # select data[j] that belongs to j-th center
            centers_index = result == j
            items = data[centers_index]

            # count the number of samples that belong to j-th center
            centers_counter_tmp = np.count_nonzero(centers_index)

            # compute the new centers
            if centers_counter_tmp:  # overflow for 0
                centers[j] = np.sum(items, axis=0) / centers_counter_tmp

    return result, centers


def kmeans_basic_twoParty_batch(dataL, dataR, k, centers_prev, centers_accumulator, centers_counter):
    """
    k: the number of centers, int
    data: {array-like, sparse matrix} of shape m x n
        m is the number of samples
        n is the number of attributes/features
    Description: plaintext KMeans for each batch
        assign data to two parties
        supporting large dataset by batch and batch
    Remark: just for testing; NOT reveal in business product
    """

    data = np.append(dataL, dataR, axis=1)
    rows, columns = np.shape(data)

    dist = np.square(np.repeat(data, k, axis=0).reshape(rows, k, columns) - centers_prev)
    distance = np.sum(dist, axis=2)
    index_min = np.argmin(distance, axis=1)
    result_batch = index_min

    for j in range(k):
        centers_index = result_batch == j
        items = data[centers_index]
        centers_counter[j] = centers_counter[j] + np.count_nonzero(centers_index)
        centers_accumulator[j] += np.sum(items, axis=0)

    # if centers_counter_tmp:  # overflow for 0
    # centers[j] = np.sum(items, axis=0) / centers_counter_tmp
    return result_batch, centers_accumulator, centers_counter


def kmeans_basic_twoParty_iteration_batches(k):
    """
    k: the number of centers, int
    data: {array-like, sparse matrix} of shape m x n
        m is the number of samples
        n is the number of attributes/features
    Description: plaintext KMeans using self-implementation
        assign data to two parties
        supporting large dataset by batch and batch
    Remark: just for testing; NOT reveal in business product
    """

    # parameter configuration and initialization
    max_iter = 2 * k
    num_samples = 8429
    batch_size = 128
    num_batches = num_samples // batch_size
    result_size = num_batches * batch_size
    result = np.empty(result_size, dtype=np.int)

    # generate data for testing
    np.random.seed(0)
    dataL = np.random.uniform(0, 10, [128, 3])
    dataR = np.random.uniform(0, 10, [128, 3])
    columns = dataL.shape[1] + dataR.shape[1]

    data = np.append(dataL, dataR, axis=1)
    centers_prev = data[np.random.choice(batch_size, k, replace=False)]

    for _ in range(max_iter):
        # refresh centers accumulator for each iteration
        centers_accumulator = np.zeros((k, columns), dtype=float)
        centers_counter = np.zeros(k, dtype=int)

        for counter_batch in range(num_batches):
            dataL = np.random.uniform(0, 10, [128, 3])
            dataR = np.random.uniform(0, 10, [128, 3])

            # compute which center that batching samples belong to and accumulate intermediate results
            counter_batch_start = counter_batch * batch_size
            counter_batch_end = counter_batch * batch_size + batch_size
            result_batch, centers_accumulator, centers_counter = kmeans_basic_twoParty_batch(dataL, dataR, k,
                                                                                             centers_prev,
                                                                                             centers_accumulator,
                                                                                             centers_counter)
            result[counter_batch_start: counter_batch_end] = result_batch

        # compute new centers for next iteration
        for index_k in range(k):
            if centers_counter[index_k] > 0:
                centers_accumulator[index_k] = centers_accumulator[index_k] / centers_counter[index_k]
        centers_prev = centers_accumulator

    return result, centers_prev


def kmeans_secure_twoParty_batch(pdataL, pdataR, k, centers_prev, centers_accumulator):
    """
    pdataL: private stf tensor of shape [n, m_1]
    pdataR: private stf tensor of shape [n, m_2]
        m = m_1 + m_2
    k: the number of centers, int
    centers_prev: a matrix, private stf tensor of shape [k, m]
    centers_accumulator: a matrix, private stf tensor of shape [k, m]
    Description: partially secure KMeans for each batch
        protect individual samples and their values, only reveal the statistical results
        assign data to two parties
        supporting large dataset by batch and batch
    Remark: just for testing; NOT reveal in business product
    """

    pdata_batch = concat([pdataL, pdataR], axis=1)

    # Compute distance to each center
    dist = (SharedPair.from_SharedPairBase(pdata_batch).expend_dims(axis=1) - centers_prev) ** 2
    distance = dist.reduce_sum(axis=2, keepdims=False)

    # num_samples = pdata_batch.shape[1]
    # Look for the nearest center of each data point
    index_min = argmin(distance, axis=1, module=None, return_min=False).to_tf_tensor('R')
    index_min = tf.squeeze(index_min, axis=1)

    result_batch = tf.cast(index_min, tf.int64)
    result_batch_oneHot = tf.one_hot(result_batch, depth=k, axis=-1)
    result_batch_oneHot = tf.cast(result_batch_oneHot, tf.int64)

    init_op = tf.compat.v1.global_variables_initializer()
    sess2 = tf.compat.v1.Session("grpc://0.0.0.0:8887")
    sess2.run(init_op)

    items_sum = (SharedPair.from_SharedPairBase(pdata_batch).transpose() @ result_batch_oneHot).transpose()
    sum_ops = centers_accumulator.assign(centers_accumulator + items_sum)
    centers_counter = np.sum(result_batch_oneHot, axis=0)
    sess2.run(sum_ops)

    return result_batch, centers_accumulator, centers_counter


def kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, max_iter, num_batches):
    """
    pdata_batch = [pdataL, pdataR], private stf tensor of shape [n, m]
        pdataL: private stf tensor of shape [n, m_1]
        pdataR: private stf tensor of shape [n, m_2]
        m = m_1 + m_2
    k: the number of centers, int
    max_iter: the number of interation, int
    centers_first: a matrix, private stf tensor of shape [k, m]
    num_batches: the number of batches, int
    Description: partially secure KMeans
        protect individual samples and their values, only reveal the statistical results
        assign data to two parties
        supporting large dataset by batch and batch
    Remark: reveal in business product
        (plan to be online in Aug, 2022)
    """
    num_features = pdata_batch.shape[1]

    # initialize centers at first epoch
    centers = SharedVariablePair(ownerL="L", ownerR="R", shape=[k, num_features], xL=centers_first.xL,
                                 xR=centers_first.xR, fixedpoint=centers_first.fixedpoint)

    # initialize centers_accumulator and counter at each epoch
    centers_accumulator = SharedVariablePair(ownerL="L", ownerR="R", shape=[k, num_features])
    centers_accumulator.load_from_numpy(np.zeros(shape=[k, num_features]))
    count_samples_in_centers = tf.Variable(initial_value=np.zeros(shape=[k]), dtype='int64')

    # Compute distance to each center
    dist = (SharedPair.from_SharedPairBase(pdata_batch).expend_dims(axis=1) - centers) ** 2
    distance = dist.reduce_sum(axis=2, keepdims=False)
    # num_samples = pdata_batch.shape[1]

    # Look for the nearest center of each data point
    index_min = argmin(distance, axis=1, module=None, return_min=False).to_tf_tensor('R')
    index_min = tf.squeeze(index_min, axis=1)
    index_min = tf.cast(index_min, tf.int64)
    index_min_oneHot = tf.one_hot(index_min, depth=k, axis=-1)
    index_min_oneHot = tf.cast(index_min_oneHot, tf.int64)

    # accumulation for each epoch
    centers_accumulator_batch = (SharedPair.from_SharedPairBase(pdata_batch).transpose() @ index_min_oneHot).transpose()
    centers_sum_ops = centers_accumulator.assign(centers_accumulator + centers_accumulator_batch)
    index_min_oneHot_sum = tf.reduce_sum(index_min_oneHot, axis=0)
    count_ops = count_samples_in_centers.assign(count_samples_in_centers + index_min_oneHot_sum)

    tmp1 = (1 - tf.cast(tf.equal(count_samples_in_centers, 0), 'float32'))  # count non-zeros
    coef1 = 1.0 / (1E-12 + tf.cast(count_samples_in_centers, 'float32')) * tmp1  # empirical results

    # compute new centers
    tmp2 = tf.expand_dims(coef1, axis=1) * centers_accumulator
    tmp3 = tf.expand_dims(tf.cast(tf.equal(count_samples_in_centers, 0), 'int64'), axis=1) * centers  # exception: count 0
    new_centers = tmp2 + tmp3
    centers_up_op = centers.assign(0.0 * centers + new_centers)

    with tf.control_dependencies([centers_sum_ops, count_ops, centers_up_op]):
        count_samples_zeros_op = count_samples_in_centers.assign(tf.zeros_like(count_samples_in_centers))
        centers_accumulator_zeros_op = centers_accumulator.assign(centers_accumulator.zeros_like())
    sess = tf.compat.v1.Session(target=StfConfig.target)
    sess.run(tf.compat.v1.initialize_all_variables())

    for i in range(max_iter):
        for j in range(num_batches):
            sess.run([centers_sum_ops, count_ops])
        sess.run([centers_up_op, count_samples_zeros_op, centers_accumulator_zeros_op])

    result = None
    for _ in range(num_batches):
        if result is None:
            result = sess.run(index_min)
        else:
            result = np.concatenate([result, sess.run(index_min)], axis=0)

    centers_np = sess.run(centers.to_tf_tensor("R"))

    return result, centers_np

start_local_server(config_file="../conf/config.json")
# data = np.array([[1.0, 2, 3, 4], [2, 3, 4, 6], [3, 3, 3, 3]])
np.random.seed(0)
np.set_printoptions(threshold=np.inf)

dataL = np.random.uniform(0, 10, [1280, 3])
dataR = np.random.uniform(0, 10, [1280, 3])
data = np.append(dataL, dataR, axis=1)
rows = dataL.shape[0]
StfConfig.default_fixed_point = 16
epoch = 10
k = 4

#  centers = data[np.random.choice(rows, k, replace=False)]
centers = data[0:k, :]

result1 = kmeans_basic(data, k, centers, epoch)
print("result1", result1)


# result2 = sKmeans.kmeans_basic_twoParty(dataL, dataR, 2, centers)
# print("result2", result2)

# result3 = sKmeans.kmeans_basic_twoParty_iteration_batches(k)
# print("result3", result3)
# np.set_printoptions(threshold=np.inf)

# start_local_server(config_file="../conf/config.json")

# parameters configuration for loading dataset: load_from_file()

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

    result4 = kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, epoch, num_batches)
    print("result4", result4)


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

    # The max number of iterations can be defined by user, e.g., 2 * k
    max_iter = 20  # 2 * k.

    # result_size = num_batches * batch_size
    # result = np.empty(result_size, dtype=np.int)  # store the classification results of each data point

    # party L: initialize PrivateTensor, load data batch by batch
    pdata_first_center_L = PrivateTensor(owner='L')
    # pdata_first_center_L.load_from_file(path=StfConfig.train_file_onL,
    #                       record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=match_col,
    #                       clip_value=clip_value)
    # pdata_first_center_L.load_from_numpy(np.random.random([k, num_features_L]))
    pdata_first_center_L.load_first_k_lines_from_file(path=StfConfig.train_file_onL, k=k, col_num=num_features_L
                                                      , sep=",", skip_row_num=1)

    # party R: initialize PrivateTensor, load data batch by batch
    pdata_first_center_R = PrivateTensor(owner='R')
    # pdata_first_center_R.load_from_file(path=StfConfig.pred_file_onR,
    #                           record_defaults=format_y, batch_size=batch_size, repeat=2, skip_col_num=match_col,
    #                           clip_value=clip_value)
    # pdata_first_center_R.load_from_numpy(np.random.random([k, num_features_R+1]))
    pdata_first_center_R.load_first_k_lines_from_file(path=StfConfig.train_file_onR, k=k, col_num=num_features_R + 1
                                                      , sep=",", skip_row_num=1)

    # split pdataR_xy to (features, label)
    pdata_first_center_R, _ = pdata_first_center_R.split(size_splits=[num_features_R, 1], axis=1)

    centers_first = concat([pdata_first_center_L, pdata_first_center_R], axis=1)
    print("center_first=", centers_first)
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
    result4 = kmeans_secure_twoParty_iteration_batches(pdata_batch, centers_first, k, max_iter, num_batches)
    print("result4", result4)


if __name__ == '__main__':
    tes_kmeans_secure_twoParty_iteration_batches0()
    # tes_kmeans_secure_twoParty_iteration_batches()
