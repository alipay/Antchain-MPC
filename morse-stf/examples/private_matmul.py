#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : private_matmul
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/6/23 下午8:14
   Description : description what the main function of this file
"""


import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.engine.start_server import start_local_server, start_client
from stensorflow.random.random import random_init
from stensorflow.homo_enc.homo_enc import homo_init

"""
A example to compute a matrix multiply of private matrix in two parties.
In the actual application, the data x and y should be
read from files in hard disk of parties using tensorflow API. For example:
with tf.device(StfConfig.workerL[0]):
    data = tf.compat.v1.data.TextLineDataset(pathL).skip(skip_row_num)
    data_iter = data.make_one_shot_iterator()
    data = data_iter.get_next()
    data = tf.reshape(data, [batch_size])
    data = tf.strings.split(data, sep=field_delim).to_tensor(default_value="0.0")
    data = data[:, skip_col_num:]
    data = tf.reshape(data, [batch_size, output_col_num])
    x = tf.strings.to_number(data, out_type='float64')
with tf.device(StfConfig.workerR[0]):
    data = tf.compat.v1.data.TextLineDataset(pathL).skip(skip_row_num)
    data_iter = data.make_one_shot_iterator()
    data = data_iter.get_next()
    data = tf.reshape(data, [batch_size])
    data = tf.strings.split(data, sep=field_delim).to_tensor(default_value="0.0")
    data = data[:, skip_col_num:]
    data = tf.reshape(data, [batch_size, output_col_num])
    y = tf.strings.to_number(data, out_type='float64')
"""

start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

x = PrivateTensor(owner='L')
x.load_from_tf_tensor(tf.constant([[0.1, 0.2], [-1.1, 0.9]]))
y = PrivateTensor(owner="R")
y.load_from_tf_tensor(tf.constant([[1.0, 2.1], [-1.4, -2.5]]))

z = x @ y
tf_z = z.to_tf_tensor("R")

with tf.compat.v1.Session(StfConfig.target) as sess:
    random_init(sess)   # initialize the random model
    print(sess.run(tf_z))
