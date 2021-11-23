#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_DNN
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-14 14:57
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.networks.DNN import DNN
import tensorflow as tf
from stensorflow.random.random import random_init
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client
import time

"""
A Example of training a DNN model (fully connected neural network) on a dataset of feature number 291 and predict using 
this model.
The features are in the party L, the label is in the party R.
"""

start_local_server(config_file="../conf/config_ym.json")
# start_client(config_file="../conf/config_ym.json", job_name="workerR")

matchColNum = 2
featureNumL = 291
featureNumR = 0
record_num = 37530

epoch = 1
batch_size = 128
learning_rate = 0.01
clip_value = 5.0
train_batch_num = epoch * record_num // batch_size + 1

pred_record_num = 12042 * 3 // 10
pred_batch_num = pred_record_num // batch_size + 1

format_x = [["a"]] * matchColNum + [[0.2]] * featureNumL
format_y = [["a"]] * matchColNum + [[1.0]]

num_features = featureNumL + featureNumR
dense_dims = [num_features, 7, 7, 1]                # the neural network structure is 7, 7, 1
l2_regularization = 0.0

# -------------define a private tensor x_train of party L and a private tensor y_train on the party R

x_train = PrivateTensor(owner='L')
y_train = PrivateTensor(owner='R')

# -----------------  load data from files -------------------
x_train.load_from_file(path=StfConfig.train_file_onL,
                       record_defaults=format_x, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                       clip_value=clip_value)

y_train.load_from_file(path=StfConfig.train_file_onR,
                       record_defaults=format_y, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                       clip_value=clip_value)

# ----------- build a DNN model (fully connected neural network)---------------

model = DNN(feature=x_train, label=y_train, dense_dims=dense_dims)

model.compile()

# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)
init_op = tf.compat.v1.initialize_all_variables()
sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()
model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization, sess=sess)
end_time = time.time()
print("train_time=", end_time - start_time)

# ------------define the private tensors for test dataset ----------------
x_test = PrivateTensor(owner='L')
y_test = PrivateTensor(owner='R')

x_test.load_from_file(path=StfConfig.pred_file_onL,
                      record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=matchColNum,
                      clip_value=clip_value)

id = y_test.load_from_file_withid(path=StfConfig.pred_file_onR,
                                  record_defaults=format_y, batch_size=batch_size, repeat=2,
                                  id_col_num=matchColNum, clip_value=clip_value)

# --------------predict --------------
model.predict_to_file(sess=sess, x=x_test, predict_file_name=StfConfig.predict_to_file,
                      batch_num=pred_batch_num, idx=id)

sess.close()
