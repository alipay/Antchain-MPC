#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : GCN_train_and_predit
   Author : qizhi.zqz
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/5/12 下午5:00
   Description : description what the main function of this file
"""
import dgl
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from stensorflow.ml.nn.networks.GCN import GCN
from stensorflow.random.random import random_init
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server, start_client
import time

dgl.backend.set_default_backend('tensorflow')
dataset = dgl.data.CoraGraphDataset()
# print('Number of categories:', dataset.num_classes)
g = dataset[0]

print("g=", g)
print("Node feature")
feature = g.ndata['feat'].numpy()
label = g.ndata['label'].numpy()
label = np.reshape(label, [-1, 1])
label = OneHotEncoder(sparse=False).fit_transform(label)
print("label=", label)

train_mask = g.ndata['train_mask'].numpy()
print("train_mask=", train_mask)
print("train set:", sum(train_mask))
print("test set:", sum(1 - train_mask))
adjacency_matrix_numpy = tf.sparse.to_dense(tf.sparse.reorder(g.adjacency_matrix())).numpy()
num_features = feature.shape[1]
# featureNumR = 0
record_num = g.num_nodes()


start_local_server(config_file="../conf/config.json")
# start_client(config_file="../conf/config_ym.json", job_name="workerR")

StfConfig.default_fixed_point = 18

StfConfig.softmax_iter_num = 32

epoch = 300
learning_rate = 0.002

dense_dims = [num_features, 16, 7]  # the neural network structure is 16, 7
l2_regularization = 0.0

# -------------define a private tensor x_train of party L and a private tensor y_train on the party R
adjacency_matrix = PrivateTensor(owner="L")
x = PrivateTensor(owner='R')
y = PrivateTensor(owner='R')

# -----------------  load data from files -------------------

x.load_from_numpy(feature)
adjacency_matrix.load_from_numpy(adjacency_matrix_numpy)
y.load_from_numpy(label)
print("y=", y)

# ----------- build a DNN model (fully connected neural network)---------------

model = GCN(feature=x, label=y, dense_dims=dense_dims,
            adjacency_matrix=adjacency_matrix, train_mask=train_mask,
            loss='CrossEntropyLossWithSoftmax')

model.compile()

# -------------start a tensorflow session, and initialize all variables -----------------
sess = tf.compat.v1.Session(StfConfig.target)
init_op = tf.compat.v1.initialize_all_variables()
sess.run(init_op)

# -------------train the model ------------------------
start_time = time.time()
# model.train_sgd(sess=sess, batch_num=epoch, learning_rate=learning_rate, l2_regularization=l2_regularization)
model.train_adam(sess=sess, batch_num=epoch, learningRate=learning_rate, eps=2E-4)
end_time = time.time()
print("train_time=", end_time - start_time)

# ------------define the private tensors for test dataset ----------------
x_test = PrivateTensor(owner='L')

# # --------------predict --------------
model.predict_to_file(sess=sess, predict_file_name=StfConfig.predict_to_file,
                      idx=tf.reshape(tf.strings.as_string(tf.range(start=0, limit=record_num)), [-1, 1])
                      , single_out=True, with_sigmoid=False)
#
sess.close()
