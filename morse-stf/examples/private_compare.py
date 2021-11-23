#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
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
from stensorflow.engine.float_compute import compare_float_float
from stensorflow.engine.start_server import start_local_server, start_client
from stensorflow.random.random import random_init
start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

"""
A Example to solve the Millionaires' Problem. 
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

x = tf.constant([[0.1, 0.2], [-1.1, 0.9]])
y = tf.constant([[1.0, 2.1], [-1.4, -2.5]])

tf_z = compare_float_float(x_owner="L", x=x, y_owner="R", y=y,
                           relation="less")
with tf.compat.v1.Session(StfConfig.target) as sess:
    random_init(sess)
    print(sess.run(tf_z.to_tf_tensor("R")))
