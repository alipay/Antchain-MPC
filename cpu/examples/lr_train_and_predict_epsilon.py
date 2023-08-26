#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : lr_train_and_predict.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/5/21 上午10:13
   Description : description what the main function of this file
"""
from stensorflow.engine.start_server import start_local_server, start_client
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.ml.logistic_regression import LogisticRegression
import numpy as np
import random
import time

random.seed(0)

"""
A Example of training a LR model on a dataset of feature number 291 and predict using 
this model.
The features are in the party L, the label is in the party R.
"""

# start_local_server(config_file="../conf/config_ym.json")
start_local_server(config_file="../conf/config_epsilon.json")
# start_client(config_file="../conf/config_ym.json", job_name="workerR")

matchColNum = 0
featureNumX = 3000
featureNumY = 0
record_num = 10

epoch = 100
batch_size = 2
learning_rate = 0.01
clip_value = 5.0
train_batch_num = epoch * record_num // batch_size + 1

pred_record_num = 10
pred_batch_num = pred_record_num // batch_size + 1

# -------------define a private tensor x_train of party L and a private tensor y_train on the party R
x_train = PrivateTensor(owner='L')
y_train = PrivateTensor(owner='R')

format_x = [["a"]] * matchColNum + [[0.2]] * featureNumX
format_y = [["a"]] * matchColNum + [[0.3]] * featureNumY + [[1.0]]

# -----------------  load data from files -------------------
x_train.load_from_file(path=StfConfig.train_file_onL,
                       record_defaults=format_x, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                       clip_value=clip_value, skip_row_num=0)

y_train.load_from_file(path=StfConfig.train_file_onR,
                       record_defaults=format_y, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                       clip_value=clip_value, skip_row_num=0)

print("StfConfig.parties=", StfConfig.parties)
# ----------- build a LR model ---------------
model = LogisticRegression(num_features=featureNumX + featureNumY, learning_rate=learning_rate)

# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)

init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()

model.fit(sess=sess, x=x_train, y=y_train, num_batches=train_batch_num)

print("train time=", time.time() - start_time)
save_op = model.save(model_file_path="./")
sess.run(save_op)

# ------------define the private tensors for test dataset ----------------
x_test = PrivateTensor(owner='L')
y_test = PrivateTensor(owner='R')

x_test.load_from_file(path=StfConfig.pred_file_onL,
                      record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=matchColNum,
                      clip_value=clip_value, skip_row_num=0)
id = y_test.load_from_file_withid(path=StfConfig.pred_file_onR,
                                  record_defaults=format_y, batch_size=batch_size, repeat=2,
                                  id_col_num=matchColNum, clip_value=clip_value, skip_row_num=0)

# --------------predict --------------
model.predict(id, x_test, pred_batch_num, sess)
sess.close()
