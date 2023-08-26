#!/usr/bin/env python
# coding=utf-8

"""
   Ant Group
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : secure_k_means.py
   Author : Yu Zheng
   Email: yuzheng0404@gmail.com
   Create Time : 2022-08-03 17:12
   Description : secure K-means
"""

import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from stensorflow.engine.start_server import start_local_server
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class import private
from stensorflow.basic.operator.algebra import concat
from stensorflow.basic.operator.argmax import argmin
from stensorflow.basic.basic_class.pair import SharedPair, SharedVariablePair
from sklearn.cluster import KMeans
import pandas as pd
from stensorflow.random.random import random_init


class SecureKMeans:
    def __init__(self, num_features, k, centers_first):
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
        self.centers = SharedVariablePair(ownerL="L", ownerR="R", shape=[k, num_features], xL=centers_first.xL,
                                     xR=centers_first.xR, fixedpoint=centers_first.fixedpoint)

        # initialize centers_accumulator and counter at each epoch
        self.centers_accumulator = SharedVariablePair(ownerL="L", ownerR="R", shape=[k, num_features])
        self.centers_accumulator.load_from_numpy(np.zeros(shape=[k, num_features]))
        self.count_samples_in_centers = tf.Variable(initial_value=np.zeros(shape=[k]), dtype='int64')
        self.k = k

    def get_train_ops(self, pdata_batch):
        # Compute distance to each center
        dist = (SharedPair.from_SharedPairBase(pdata_batch).expend_dims(axis=1) - self.centers) ** 2
        distance = dist.reduce_sum(axis=2, keepdims=False)
        # num_samples = pdata_batch.shape[1]

        # Look for the nearest center of each data point
        index_min = argmin(distance, axis=1, module=None, return_min=False).to_tf_tensor('R')
        index_min = tf.squeeze(index_min, axis=1)
        index_min = tf.cast(index_min, tf.int64)
        index_min_oneHot = tf.one_hot(index_min, depth=self.k, axis=-1)
        index_min_oneHot = tf.cast(index_min_oneHot, tf.int64)

        # accumulation for each epoch
        centers_accumulator_batch = (
                    SharedPair.from_SharedPairBase(pdata_batch).transpose() @ index_min_oneHot).transpose()
        centers_sum_ops = self.centers_accumulator.assign(self.centers_accumulator + centers_accumulator_batch)
        index_min_oneHot_sum = tf.reduce_sum(index_min_oneHot, axis=0)
        count_ops = self.count_samples_in_centers.assign(self.count_samples_in_centers + index_min_oneHot_sum)

        tmp1 = (1 - tf.cast(tf.equal(self.count_samples_in_centers, 0), 'float32'))  # count non-zeros
        coef1 = 1.0 / (1E-12 + tf.cast(self.count_samples_in_centers, 'float32')) * tmp1  # empirical results

        # compute new centers
        tmp2 = tf.expand_dims(coef1, axis=1) * self.centers_accumulator
        tmp3 = tf.expand_dims(tf.cast(tf.equal(self.count_samples_in_centers, 0), 'int64'),
                              axis=1) * self.centers  # exception: count 0
        new_centers = tmp2 + tmp3
        centers_up_op = self.centers.assign(0.0 * self.centers + new_centers)

        with tf.control_dependencies([centers_sum_ops, count_ops, centers_up_op]):
            count_samples_zeros_op = self.count_samples_in_centers.assign(tf.zeros_like(self.count_samples_in_centers))
            centers_accumulator_zeros_op = self.centers_accumulator.assign(self.centers_accumulator.zeros_like())

        return centers_sum_ops, count_ops, centers_up_op, count_samples_zeros_op, centers_accumulator_zeros_op

    def train(self, pdata_batch, epoch, batchs_in_epoch, sess):
        centers_sum_ops, count_ops, centers_up_op, count_samples_zeros_op, centers_accumulator_zeros_op = self.get_train_ops(pdata_batch)
        sess.run(tf.compat.v1.initialize_all_variables())
        for i in range(epoch):
            for j in range(batchs_in_epoch):
                print("epoch {}, batch {}".format(i, j))
                sess.run([centers_sum_ops, count_ops])
            sess.run([centers_up_op, count_samples_zeros_op, centers_accumulator_zeros_op])

    def save(self, sess, model_file_machine="R", path="./output/model"):
        print("save model...")
        centers_np = sess.run(self.centers.to_tf_tensor(model_file_machine))
        centers_pd = pd.DataFrame(data=centers_np)
        centers_pd.to_csv(path, header=False, index=False)

    def load(self, path="./output/model/"):
        print("load model...")
        value = pd.read_csv(path, header=None, index_col=None)
        value = np.array(value)
        value = np.reshape(value, self.centers.shape)
        self.centers.load_from_numpy(value, const=True)

    def predict_batch(self, pdata_batch):
        dist = (SharedPair.from_SharedPairBase(pdata_batch).expend_dims(axis=1) - self.centers) ** 2
        distance = dist.reduce_sum(axis=2, keepdims=False)
        # num_samples = pdata_batch.shape[1]

        # Look for the nearest center of each data point
        index_min = argmin(distance, axis=1, module=None, return_min=False).to_tf_tensor('R')
        index_min = tf.squeeze(index_min, axis=1)
        index_min = tf.cast(index_min, tf.int64)
        return index_min

    def predict_to_file(self, sess, pdata_batch, predict_file_name,
                        batch_num, idx, model_file_machine="R", record_num_ceil_mod_batch_size=0):
        print("predict_file_name=", predict_file_name)
        index_pred = self.predict_batch(pdata_batch)
        sess.run(tf.compat.v1.initialize_all_variables())
        index_pred = tf.reshape(index_pred, [-1,1])
        #id_y_pred = index_pred.to_tf_str(owner=model_file_machine, id_col=idx)
        y = tf.strings.as_string(index_pred)
        id_y = tf.concat([idx, y], axis=1)
        id_y = tf.compat.v1.reduce_join(id_y, separator=",", axis=-1)

        random_init(sess)

        with open(predict_file_name, "w") as f:
            for batch in range(batch_num - 1):
                records = sess.run(id_y)
                records = "\n".join(records.astype('str'))
                f.write(records + "\n")

            records = sess.run(id_y)[0:record_num_ceil_mod_batch_size]
            records = "\n".join(records.astype('str'))
            f.write(records + "\n")
