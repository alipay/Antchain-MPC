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

import tensorflow as tf
from stensorflow.global_var import StfConfig
import random
import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
random.seed(0)
"""
A Example of training a LR model on a dataset of feature number 291 and predict using 
this model.
The features are in the party L, the label is in the party R.
"""

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
train_file_onL = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x"
train_file_onR = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_y"
pred_file_onL = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x"
pred_file_onR = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_y"
pred_file = "./predict"


format_x = [["a"]] * matchColNum + [[0.2]] * featureNumX
format_y = [["a"]] * matchColNum + [[0.3]] * featureNumY + [[1.0]]


def load_from_file(path: str, record_defaults, batch_size, field_delim=",", skip_row_num=1, skip_col_num=0,
                   repeat=1, clip_value=None, scale=1.0, map_fn=None, output_col_num=None, buffer_size=0):
    """
    Load data from file
    :param path:  absolute path of file in the disk of self.owner
    :param record_defaults: for example [['a'], ['a'], [1.0], [1.0], [1.0]]
    :param batch_size:
    :param field_delim:    field delim between columns
    :param skip_row_num:   skip row number in head of the file
    :param skip_col_num:   skip column number in the file
    :param repeat:         repeat how many times of the file
    :param clip_value:     the features are clip by this value such that |x|<=clip_value
    :param scale:          multiply scale for the  columns
    :param map_fn:         A map function for the columns, for example: lambda x: x[3]*x[4]
    :param output_col_num:   output column number
    :param buffer_size:       buffer size
    :return:
    """

    def clip(r):
        if clip_value is None:
            return r * scale if scale != 1.0 else r
        else:
            return tf.clip_by_value(r * scale, -clip_value, clip_value)

    if output_col_num is None:
        output_col_num = len(record_defaults) - skip_col_num


    data = tf.compat.v1.data.TextLineDataset(path, buffer_size=buffer_size).skip(skip_row_num)
    data_iter = data.repeat(repeat).batch(batch_size).make_one_shot_iterator()
    data = data_iter.get_next()
    data = tf.reshape(data, [batch_size])
    data = tf.strings.split(data, sep=field_delim).to_tensor(default_value="0.0")
    data = data[:, skip_col_num:]
    data = tf.reshape(data, [batch_size, output_col_num])
    data = tf.strings.to_number(data, out_type='float64')
    data = clip(data)
    if map_fn is not None:
        data = data.map(map_func=map_fn)
    return data
# -----------------  load data from files -------------------
x_train=load_from_file(path=train_file_onL,
                        record_defaults=format_x, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                        clip_value=clip_value, skip_row_num=0)

y_train=load_from_file(path=train_file_onR,
                         record_defaults=format_y, batch_size=batch_size, repeat=epoch + 2, skip_col_num=matchColNum,
                         clip_value=clip_value, skip_row_num=0)

print("StfConfig.parties=", StfConfig.parties)
# ----------- build a LR model ---------------

model = Sequential()
model.add(Dense(units=1, activation='sigmoid', input_dim=featureNumX))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

for _ in range(train_batch_num):
    if _%10 == 0:
        print("train step", _)
    model.train_on_batch(x_train, y_train)


x_test = load_from_file(path=pred_file_onL,
                       record_defaults=format_x, batch_size=batch_size, repeat=2, skip_col_num=matchColNum,
                       clip_value=clip_value, skip_row_num=0)
y_test = load_from_file(path=pred_file_onR,
                                    record_defaults=format_y, batch_size=batch_size, repeat=2,
                                    skip_col_num=matchColNum, clip_value=clip_value, skip_row_num=0)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128, steps=pred_batch_num)


with open(pred_file, "w") as f:
    for _ in range(pred_batch_num):
        if _%10 == 0:
            print("pred step ", _)
        y_pred = model.predict_on_batch(x_test)
        for q in y_pred:
            #print(y_pred)
            f.write("{}\n".format(q[0]))
# --------------predict --------------
# model.predict(id, x_test, pred_batch_num, sess)
# sess.close()
#
