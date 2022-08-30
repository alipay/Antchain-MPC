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

from stensorflow.ml.nn.networks.DNN_with_SL import DNN_with_SL2
import tensorflow as tf
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.global_var import StfConfig
from stensorflow.random.random import random_init
from stensorflow.engine.start_server import start_local_server, start_client
import time

"""
A Example of training a DNN model (fully connected neural network) with security level on a dataset of feature number 10 and predict using 
this model.
The features f0 ~ f4 are in the party L, the features f5 ~ f9 are in the party R, the label is in the party R.
"""

start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config.json", job_name="workerR")

matchColNum = 1
featureNumL = 5  # feature number in workerL
featureNumR = 5  # feature number in workerR
record_num = 12042 * 7 // 10  # 30000 #50038
epoch = 1
batch_size = 128
l2_regularization = 0.0
clip_value = 5.0
batch_num_per_epoch = record_num // batch_size
train_batch_num = epoch * batch_num_per_epoch + 1
learning_rate = 0.01

# -------------define a private tensor x_train of party L and a private tensor xyR_train on the party R
xL_train = PrivateTensor(owner='L')  # features that is private data in workerL
xyR_train = PrivateTensor(owner='R')  # features and label that is  private data in workerR

format_x = [["a"]] * matchColNum + [[0.2]] * featureNumL  # data format inn workerL
format_y = [["a"]] * matchColNum + [[0.3]] * featureNumR + [[1.0]]  # data format in workerR

# -----------------  load data from files -------------------

xL_train.load_from_file(path=StfConfig.train_file_onL, record_defaults=format_x, batch_size=batch_size,
                        repeat=epoch + 2, skip_col_num=matchColNum, clip_value=clip_value)

xyR_train.load_from_file(path=StfConfig.train_file_onR, record_defaults=format_y, batch_size=batch_size,
                         repeat=epoch + 2, skip_col_num=matchColNum, clip_value=clip_value)

# split xyR_train to features xR_train and label y_train
xR_train, y_train = xyR_train.split(size_splits=[featureNumR, 1], axis=1)

num_features = featureNumL + featureNumR

# ----------- build a DNN model (fully connected neural network)---------------
model = DNN_with_SL2(feature=xL_train, label=y_train, dense_dims=[num_features, 32, 32, 1],
                     feature_another=xR_train, secure_config=[2, 1, 1])
# A DNN model with dense layers of dimensions 32, 32, 1 and secure_config=[2, 1, 1]

model.compile()
# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)
init_op = tf.compat.v1.initialize_all_variables()
sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()
model.train_sgd(learning_rate=learning_rate, batch_num=train_batch_num, l2_regularization=l2_regularization, sess=sess)
# train the model
end_time = time.time()

print("train_time=", end_time - start_time)
model.save(sess=sess, path="../output/model")

model.load(path="../output/model")

# ------------define the private tensors for test dataset ----------------
pred_record_num = 12042 * 3 // 10
pred_batch_num = pred_record_num // batch_size

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
y_pred = model.predict(xL_test, xR_test, with_sigmoid=False)

id_y_pred = y_pred.to_tf_str(owner="R", id_col=id)

random_init(sess)

with open(StfConfig.predict_to_file, "w") as f:
    for batch in range(pred_batch_num):
        records = sess.run(id_y_pred)
        records = "\n".join(records.astype('str'))
        f.write(records + "\n")
sess.close()
