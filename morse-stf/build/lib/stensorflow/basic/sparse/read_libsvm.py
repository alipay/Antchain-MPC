#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : read_libsvm
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/3/22 下午3:10
   Description : description what the main function of this file
"""
import tensorflow as tf

def decode_libsvm(line):
    columns = tf.compat.v1.string_split([line], ' ')
    labels = tf.compat.v1.string_to_number(columns.values[0], out_type=tf.float32)
    splits = tf.compat.v1.string_split(columns.values[1:], ':')
    print("splits=",splits)
    """
    SparseTensorValue(indices=array([[ 0,  0],
       [ 0,  1],
       [ 1,  0],
       [ 1,  1],
       [ 2,  0],
       [ 2,  1],
       [ 3,  0],
       [ 3,  1],
       [ 4,  0],
       [ 4,  1],
       [ 5,  0],
       [ 5,  1],
       [ 6,  0],
       [ 6,  1],
       [ 7,  0],
       [ 7,  1],
       [ 8,  0],
       [ 8,  1],
       [ 9,  0],
       [ 9,  1],
       [10,  0],
       [10,  1],
       [11,  0],
       [11,  1],
       [12,  0],
       [12,  1],
       [13,  0],
       [13,  1]]), values=array([b'3', b'1', b'11', b'1', b'14', b'1', b'19', b'1', b'39', b'1',
       b'42', b'1', b'55', b'1', b'64', b'1', b'67', b'1', b'73', b'1',
       b'75', b'1', b'76', b'1', b'80', b'1', b'83', b'1'], dtype=object), dense_shape=array([14,  2]))
    """
    id_vals = tf.reshape(splits.values,splits.dense_shape)
    print("id_vals=", id_vals)
    """
    [[b'3' b'1']
 [b'11' b'1']
 [b'14' b'1']
 [b'19' b'1']
 [b'39' b'1']
 [b'42' b'1']
 [b'55' b'1']
 [b'64' b'1']
 [b'67' b'1']
 [b'73' b'1']
 [b'75' b'1']
 [b'76' b'1']
 [b'80' b'1']
 [b'83' b'1']]
    """
    feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
    feat_ids = tf.compat.v1.string_to_number(feat_ids, out_type=tf.int32)
    feat_ids = tf.squeeze(feat_ids, axis=-1)
    feat_vals = tf.compat.v1.string_to_number(feat_vals, out_type=tf.float32)
    feat_vals = tf.squeeze(feat_vals, axis=-1)
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
    #return id_vals



def decode_libsvm_x(line, field_delim=" ", skip_col_num=0):
    columns = tf.compat.v1.string_split([line], field_delim)
    #labels = tf.compat.v1.string_to_number(columns.values[0], out_type=tf.float32)
    splits = tf.compat.v1.string_split(columns.values[skip_col_num:], ':')
    id_vals = tf.reshape(splits.values,splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2,axis=1)
    feat_ids = tf.compat.v1.string_to_number(feat_ids, out_type=tf.int64)
    # feat_ids = tf.squeeze(feat_ids, axis=-1)
    feat_vals = tf.compat.v1.string_to_number(feat_vals, out_type=tf.float32)
    feat_vals = tf.squeeze(feat_vals, axis=-1)
    return feat_ids, feat_vals





# Extract lines from input files using the Dataset API, can pass one filename or filename list

def load_libsvm_x_to_dense(filenames, feature_num, batch_size, field_delim=" ", skip_row_num=1, skip_col_num=0,
                           repeat=1, clip_value=None, scale=1.0, map_fn=None, output_col_num=None, buffer_size=0):
    def decode_libsvm_x_to_dense(line):
        feat_ids, feat_vals = decode_libsvm_x(line, field_delim=field_delim, skip_col_num=skip_col_num)
        r_sparse = tf.SparseTensor(indices=feat_ids-1, values=feat_vals, dense_shape=[feature_num])
        return tf.compat.v1.sparse_tensor_to_dense(r_sparse, default_value=0.0)
    def clip(r):
        if clip_value is None:
            return r * scale if scale != 1.0 else r
        else:
            return tf.clip_by_value(r * scale, -clip_value, clip_value)
    if output_col_num is None:
        output_col_num = feature_num - skip_col_num
    dataset = tf.compat.v1.data.TextLineDataset(filenames, buffer_size=buffer_size)\
        .skip(skip_row_num).map(decode_libsvm_x_to_dense, num_parallel_calls=batch_size)\
        .prefetch(1000)

    data = dataset.repeat(repeat).batch(batch_size=batch_size).make_one_shot_iterator().get_next()
    # print("feature_num=", feature_num)
    # print("output_col_num=", output_col_num)
    data = tf.reshape(data, [batch_size, output_col_num])
    if map_fn is not None:
        data = data.map(map_func=map_fn)
    data = clip(data)
    return data

def load_libsvm_x_to_sparse(filenames, batch_size, feature_num, skip_row_num=0):
    def decode_libsvm_x_to_sparse(line):
        feat_ids, feat_vals = decode_libsvm_x(line)
        r_sparse = tf.SparseTensor(indices=feat_ids-1, values=feat_vals, dense_shape=[feature_num])
        return r_sparse
    dataset = tf.compat.v1.data.TextLineDataset(filenames).skip(skip_row_num).map(decode_libsvm_x_to_sparse, num_parallel_calls=batch_size).prefetch(1000)
    data = dataset.batch(batch_size=batch_size).make_one_shot_iterator().get_next()
    return data

def load_libsvm_x(filenames, feature_num, batch_size, field_delim=" ", skip_row_num=1, skip_col_num=0,
                           repeat=1, clip_value=None, scale=1.0, map_fn=None,
                  output_col_num=None, buffer_size=0, sparse_flag=False):
    if not sparse_flag:
        return load_libsvm_x_to_dense(filenames, feature_num, batch_size, field_delim, skip_row_num, skip_col_num,
                           repeat, clip_value, scale, map_fn, output_col_num, buffer_size)
    else:
        return load_libsvm_x_to_sparse(filenames, batch_size, feature_num)