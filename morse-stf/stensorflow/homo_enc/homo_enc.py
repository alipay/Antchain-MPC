#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : homo_enc.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/9/29 下午3:19
   Description : description what the main function of this file
"""
from stensorflow.global_var import StfConfig
import tensorflow as tf

def homo_init(sess=None):
    init_coll = tf.compat.v1.get_collection(StfConfig.coll_name_vars_homo)
    init_homo_op = tf.compat.v1.initialize_variables(init_coll)
    if sess is not None:
        sess.run(init_homo_op)
    else:
        return init_homo_op


def gene_key():
    homo_module = StfConfig.homo_module
    gene_key_zero = tf.Variable(initial_value=[0], trainable=False)
    tf.compat.v1.add_to_collection(StfConfig.coll_name_vars_homo, gene_key_zero)
    sk, pk, gk = homo_module.gen_key(gene_key_zero)
    return sk, pk, gk

def enc(pk, x):
    homo_module = StfConfig.homo_module
    return homo_module.enc(pk, x)

def mat_mul_vec_to_share(pk, gk, mat, cipher_vec):
    homo_module = StfConfig.homo_module
    return homo_module.mat_mul_vec_to_share(pk, gk, mat, cipher_vec)

def dec(sk, size, cipher_vec):
    homo_module = StfConfig.homo_module
    z = homo_module.dec(sk, size, cipher_vec)
    return tf.reshape(z, size)

def vec_mul_vec(pk, plain_vec, cipher_vec):
    homo_module = StfConfig.homo_module
    return homo_module.vec_mul_vec(pk, plain_vec, cipher_vec)



def cipher_to_share(size, pk, cipher):
    homo_module = StfConfig.homo_module
    cipher_out, share_out = homo_module.cipher_to_share(size, pk, cipher)
    share_out = tf.reshape(share_out, [size])
    return (cipher_out, share_out)


def vec_mul_vec_to_share(pk, plain_vec, cipher_vec):
    cipher_out = vec_mul_vec(pk, plain_vec, cipher_vec)
    size = tf.cast(tf.shape(plain_vec)[0],"int64")
    z = cipher_to_share(size, pk, cipher_out)
    return z
