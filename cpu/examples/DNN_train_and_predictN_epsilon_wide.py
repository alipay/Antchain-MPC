#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : lr_train_and_predict.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/4/11 下午15:51
   Description : description what the main function of this file
"""
from stensorflow.engine.start_server import start_local_server, start_client
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.ml.logistic_regression_wide import LogisticRegressionWide
from stensorflow.ml.nn.networks.DNN_with_SL import DNN_with_SLN
from stensorflow.basic.operator.algebra import concat
import numpy as np
import math
import random
import time
random.seed(0)


start_local_server(config_file="../conf/config_epsilon.json")
# start_client(config_file="../conf/config_ym.json", job_name="workerR")

skipColNums = [0, 0, 0]
featureNums = [1000, 1000, 1000]
owners = ["L", "R", "S"]
with_y = [False, True, False]
train_files_onL = ["/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/dataset/epsilon_normalized_test_x30", "", "/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/dataset/epsilon_normalized_test_x31L"]
train_files_onR = ["", "/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/dataset/epsilon_normalized_test_x32y", "/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/dataset/epsilon_normalized_test_x31R"]
test_files_onL = train_files_onL
test_files_onR = train_files_onR
dense_dim = [3000, 512, 256, 64, 1]
secure_config = [2, 0, 3]
header_bytes=8
footer_bytes=0
record_num = 4


epoch = 100
batch_size = 2
learning_rate = 0.01
clip_value = 5.0
train_batch_num = epoch * record_num // batch_size + 1

pred_record_num = 4
pred_batch_num = pred_record_num // batch_size + 1

# -------------define a private tensor x_train of party L and a private tensor y_train on the party R
x_trains = []
for party_id in range(len(featureNums)):
    if owners[party_id] != "S":
        train_file = train_files_onL[party_id] + train_files_onR[party_id]
        if with_y[party_id]:
            xy_train = PrivateTensor(owner=owners[party_id])
            xy_train.load_from_file(path=train_file,
                                     record_defaults=[1.0]*featureNums[party_id]+[1.0], batch_size=batch_size, repeat=epoch + 2,
                                     skip_col_num=skipColNums[party_id],
                                     clip_value=clip_value, skip_row_num=1)

            # split xyR_train to features xR_train and label y_train
            x_train, y_train = xy_train.split(size_splits=[featureNums[party_id], 1], axis=1)
        else:
            x_train = PrivateTensor(owner=owners[party_id])
            x_train.load_from_file(path=train_file,
                                   record_defaults=[1.0]*(featureNums[party_id]+skipColNums[party_id]), batch_size=batch_size, repeat=epoch + 2,
                                   skip_col_num=skipColNums[party_id],
                                   clip_value=clip_value, skip_row_num=1)
    else:
        x_train = SharedPair(ownerL="L", ownerR="R", shape=[batch_size, featureNums[party_id]])
        x_train.load_from_fixed_length_file(pathL=train_files_onL[party_id], pathR=train_files_onR[party_id],
                                            header_bytes=header_bytes, footer_bytes=footer_bytes,
                                            fields_num=featureNums[party_id], batch_size=batch_size, repeat=epoch + 2)
    x_trains += [x_train]

# x_train = concat(x_trains, axis=1)
# x_train = SharedPair.from_SharedPairBase(x_train)

# -----------------  load data from files -------------------

print("StfConfig.parties=", StfConfig.parties)

# ----------- build a DNN model ---------------
model = DNN_with_SLN(features=x_trains, label=y_train, dense_dims=[3000, 32, 32, 1], secure_config=[1, 1, 2])

model.compile()
# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)

init_op = tf.compat.v1.global_variables_initializer()


sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()

model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=0.0,
                sess=sess)

print("train time=", time.time()-start_time)
model.save(sess, model_file_machine="R", path=StfConfig.stf_home_workerR+"/output/model")

model.load(path=StfConfig.stf_home_workerR+"/output/model")

# ------------define the private tensors for test dataset ----------------
x_tests = []
for party_id in range(len(featureNums)):
    if owners[party_id] != "S":
        test_file = test_files_onL[party_id] + test_files_onR[party_id]
        if with_y[party_id]:
            xy_test = PrivateTensor(owner=owners[party_id])
            id = xy_test.load_from_file_withid(path=test_file,
                                              record_defaults=[1.0]*(featureNums[party_id]+1), batch_size=batch_size, repeat=2,
                                              id_col_num=skipColNums[party_id], clip_value=clip_value, skip_row_num=1)

            # split xyR_test to features xR_test and label y_test
            x_test, y_test = xy_test.split(size_splits=[featureNums[party_id], 1], axis=1)
        else:
            x_test = PrivateTensor(owner=owners[party_id])
            x_test.load_from_file(path=test_file,
                                   record_defaults=[1.0]*(featureNums[party_id]+skipColNums[party_id]), batch_size=batch_size, repeat=epoch + 2,
                                   skip_col_num=skipColNums[party_id],
                                   clip_value=clip_value, skip_row_num=1)
    else:
        x_test = SharedPair(ownerL="L", ownerR="R", shape=[batch_size, featureNums[party_id]])
        x_test.load_from_fixed_length_file(pathL=test_files_onL[party_id], pathR=test_files_onR[party_id],
                                            header_bytes=header_bytes, footer_bytes=footer_bytes,
                                           fields_num=featureNums[party_id], batch_size=batch_size, repeat=epoch+2)
    x_tests += [x_test]


# --------------predict --------------
start_time = time.time()
model.predict_to_file(sess, xs=x_tests, predict_file_name=StfConfig.predict_to_file,
                      batch_num=pred_batch_num,idx=id, model_file_machine="R",
                      record_num_ceil_mod_batch_size=int(math.ceil(pred_batch_num/batch_size))
                      ,with_sigmoid=True)
print("predict time=", time.time()-start_time)
sess.close()

