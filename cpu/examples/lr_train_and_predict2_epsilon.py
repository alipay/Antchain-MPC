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
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.random.random import random_init
from stensorflow.ml.logistic_regression2 import LogisticRegression2
import time
import random
random.seed(0)



start_local_server(config_file="../conf/config_epsilon2_450.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

matchColNum = 1
featureNumL = 450 #1000 #1500
featureNumR = 450 #1000 #1500
record_num = 2
epoch = 100  # 15
batch_size = 2
learning_rate = 0.1
clip_value = 5.0

train_batch_num = 1 # epoch * record_num // batch_size + 1
pred_record_num = 200000
pred_batch_size = 1024
pred_batch_num = pred_record_num // pred_batch_size

# -------------define a private tensor x_train of party L and a private tensor xyR_train on the party R
xL_train = PrivateTensor(owner='L')
xyR_train = PrivateTensor(owner="R")


format_x = [["a"]] * matchColNum + [[0.2]] * featureNumL
format_y = [["a"]] * matchColNum + [[0.3]] * featureNumR + [[1.0]]

# -----------------  load data from files -------------------

xL_train.load_from_file(path=StfConfig.train_file_onL,
                        record_defaults=format_x, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                        clip_value=clip_value, skip_row_num=1)
print("XL_train=", xL_train)

xyR_train.load_from_file(path=StfConfig.train_file_onR,
                         record_defaults=format_y, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                         clip_value=clip_value, skip_row_num=1)

print("xyR_train=", xyR_train)

# split xyR_train to features xR_train and label y_train
xR_train, y_train = xyR_train.split(size_splits=[featureNumR, 1], axis=1)

# ----------- build a LR model ---------------
model = LogisticRegression2(learning_rate=learning_rate, num_features_L=featureNumL, num_features_R=featureNumR)


# -------------start a tensorflow session, and initialize all variables -----------------
#sess = tf.compat.v1.Session(StfConfig.target)
sess = tf.compat.v1.Session(StfConfig.target,
    config=tf.compat.v1.ConfigProto(
    device_count={"CPU":12},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1))

init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)


# -------------train the model ------------------------
start_time = time.time()

model.fit(sess, x_L=xL_train, x_R=xR_train, y=y_train, num_batches=train_batch_num)

print("train time=", time.time() - start_time)
save_op = model.save(model_file_path="./output/model")
sess.run(save_op)
model.load(model_file_path="./output/model")


# ------------define the private tensors for test dataset ----------------
xL_test = PrivateTensor(owner='L')
xRy_test = PrivateTensor(owner='R')

xL_test.load_from_file(path=StfConfig.pred_file_onL,
                       record_defaults=format_x, batch_size=pred_batch_size, repeat=2, skip_col_num=matchColNum,
                       clip_value=clip_value)

id = xRy_test.load_from_file_withid(path=StfConfig.pred_file_onR,
                                    record_defaults=format_y, batch_size=pred_batch_size, repeat=2,
                                    id_col_num=matchColNum, clip_value=clip_value)

xR_test, y_test = xRy_test.split(size_splits=[-1, 1], axis=1)

# --------------predict --------------
print("StfConfig.predict_to_file=", StfConfig.predict_to_file)
start_time = time.time()
model.predict_simple(id, xL_test, xR_test, pred_batch_num, sess, predict_file=StfConfig.predict_to_file)
print("predict time=", time.time() - start_time)
sess.close()
