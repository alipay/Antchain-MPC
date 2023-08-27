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
import sys
sys.path.append("/Users/qizhi.zqz/projects/morse-stf")


from stensorflow.engine.start_server import start_local_server, start_client
import tensorflow as tf
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.random.random import random_init
from stensorflow.ml.logistic_regression2 import LogisticRegression2
import time
import random
random.seed(0)
"""
A Example of training a LR model on a dataset of feature number 10 and predict using 
this model.
The features f0 ~ f4 are in the party L, the features f5 ~ f9 are in the party R, the label is in the party R.
"""


#start_local_server(config_file="../conf/config_parties2.json")
start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")
# StfConfig.truncation_functionality = True

matchColNum = 1
featureNumL = 5
featureNumR = 5
record_num = 8429
epoch = 5  # 15
batch_size = 128
learning_rate = 0.1
clip_value = 5.0

train_batch_num = epoch * record_num // batch_size + 1
pred_record_num = 12042 * 3 // 10
pred_batch_size = 128
pred_batch_num = pred_record_num // pred_batch_size

# -------------define a private tensor x_train of party L and a private tensor xyR_train on the party R
xL_train = PrivateTensor(owner='L')
xyR_train = PrivateTensor(owner="R")


format_x = [["a"]] * matchColNum + [[0.2]] * featureNumL
format_y = [["a"]] * matchColNum + [[0.3]] * featureNumR + [[1.0]]

# -----------------  load data from files -------------------

xL_train.load_from_file(path=StfConfig.train_file_onL,
                        record_defaults=format_x, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                        clip_value=clip_value)

xyR_train.load_from_file(path=StfConfig.train_file_onR,
                         record_defaults=format_y, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                         clip_value=clip_value)

# split xyR_train to features xR_train and label y_train
xR_train, y_train = xyR_train.split(size_splits=[featureNumR, 1], axis=1)

# ----------- build a LR model ---------------
model = LogisticRegression2(learning_rate=learning_rate, num_features_L=featureNumL, num_features_R=featureNumR)


# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()

model.fit(sess, x_L=xL_train, x_R=xR_train, y=y_train, num_batches=train_batch_num)

print("train time=", time.time() - start_time)
save_op = model.save(model_file_path="../output")
sess.run(save_op)
model.load(model_file_path="../output")
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)


# ------------define the private tensors for test dataset ----------------
xL_test = PrivateTensor(owner='L')
xRy_test = PrivateTensor(owner='R')

xL_test.load_from_file(path=StfConfig.pred_file_onL,
                       record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=matchColNum,
                       clip_value=clip_value)

id = xRy_test.load_from_file_withid(path=StfConfig.pred_file_onR,
                                    record_defaults=format_y, batch_size=batch_size, repeat=2,
                                    id_col_num=matchColNum, clip_value=clip_value)

xR_test, y_test = xRy_test.split(size_splits=[-1, 1], axis=1)

# --------------predict --------------
print("StfConfig.predict_to_file=", StfConfig.predict_to_file)
start_time = time.time()
model.predict_simple(id, xL_test, xR_test, pred_batch_num, sess, predict_file=StfConfig.predict_to_file)
print("predict time=", time.time() - start_time)
sess.close()
