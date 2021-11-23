#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : test_homo
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2021/9/16 下午9:14
   Description : description what the main function of this file
"""

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

size=100
homo_module = tf.load_op_library("/Users/qizhi.zqz/projects/morse-stf/morse-stf/cops/_stf_homo_macos.so")
random_module = tf.load_op_library("/Users/qizhi.zqz/projects/morse-stf/morse-stf/cops/_stf_random_macos.so")

def test_mat_mul_vec():
    sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0])) # 一定要以这个tf.Variable作为入参，否则会有错误
    row_num = 300
    col_num = 100
    plaintext = random_module.rint64(shape=[col_num], step=tf.Variable(initial_value=[0]))
    #plaintext = tf.cast(tf.range(start=0, limit=size), 'int64') # (shape=[size], dtype='int64')

    cipher = homo_module.enc(pk, plaintext)
    print(cipher)

    #A = tf.constant(np.random.random_integers(-(1<<63), (1<<63)-1, size=[size,size]), dtype='int64')
    A=tf.random.uniform(shape=[row_num, col_num], minval=-(1<<63), maxval=(1<<63)-1, dtype='int64')

    plaintext_out=tf.squeeze(tf.matmul(A, tf.expand_dims(plaintext, axis=-1)))

    cipher_v_out = homo_module.mat_mul_vec(pk, gk, A, cipher)
    print("cipher_v_out=", cipher_v_out)



    plaintext_out2 = homo_module.dec(sk, [row_num], cipher_v_out)
    print(plaintext_out2)

    sess=tf.compat.v1.Session()

    sess.run(tf.compat.v1.initialize_all_variables())

    print(np.sum((sess.run(plaintext_out - plaintext_out2))**2))
    #sess=tf.compat.v1.Session()
    #print(sess.run(keys))

def test_mat_mul_vec_to_share():
    sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0]))  # 一定要以这个tf.Variable作为入参，否则会有错误
    row_num = 3000
    col_num = 128
    #plaintext = random_module.rint64(shape=[col_num], step=tf.Variable(initial_value=[0]))
    plaintext = tf.cast(tf.range(start=0, limit=col_num), 'int64') # (shape=[size], dtype='int64')

    cipher = homo_module.enc(pk, plaintext)
    print(cipher)

    # A = tf.constant(np.random.random_integers(-(1<<63), (1<<63)-1, size=[size,size]), dtype='int64')
    A = tf.random.uniform(shape=[row_num, col_num], minval=-(1 << 63), maxval=(1 << 63) - 1, dtype='int64')

    plaintext_out = tf.squeeze(tf.matmul(A, tf.expand_dims(plaintext, axis=-1)))

    plaintext_out1, cipher_v_out = homo_module.mat_mul_vec_to_share(pk, gk, A, cipher)
    print("cipher_v_out=", cipher_v_out)

    plaintext_out2 = homo_module.dec(sk, [row_num], cipher_v_out)
    print(plaintext_out2)

    sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.initialize_all_variables())

    print(np.sum((sess.run(plaintext_out - (plaintext_out1+plaintext_out2) )) ** 2))


def test_mat_mul_mat_to_share():
    sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0]))  # 一定要以这个tf.Variable作为入参，否则会有错误

    plaintext = random_module.rint64(shape=[2,3,4, size,1], step=tf.Variable(initial_value=[0]))
    # plaintext = tf.cast(tf.range(start=0, limit=size), 'int64') # (shape=[size], dtype='int64')
    cipher = homo_module.enc(pk, plaintext)
    print(cipher)

    # A = tf.constant(np.random.random_integers(-(1<<63), (1<<63)-1, size=[size,size]), dtype='int64')
    A = tf.random.uniform(shape=[2,3,4, size, size], minval=-(1 << 63), maxval=(1 << 63) - 1, dtype='int64')

    plaintext_out = tf.squeeze(tf.matmul(A, tf.expand_dims(plaintext, axis=-1)))

    plaintext_out1, cipher_v_out = homo_module.mat_mul_vec_to_share(pk, gk, A, cipher)
    print("cipher_v_out=", cipher_v_out)

    plaintext_out2 = homo_module.dec(sk, [size], cipher_v_out)
    print(plaintext_out2)

    sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.initialize_all_variables())

    print(np.sum((sess.run(plaintext_out - (plaintext_out1+plaintext_out2) )) ** 2))


def test_vec_mul_vec():
    sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0])) # 一定要以这个tf.Variable作为入参，否则会有错误
    col_num = 100
    plaintext = random_module.rint64(shape=[col_num], step=tf.Variable(initial_value=[0]))
    #plaintext = tf.cast(tf.range(start=0, limit=size), 'int64') # (shape=[size], dtype='int64')

    cipher = homo_module.enc(pk, plaintext)
    print(cipher)

    #A = tf.constant(np.random.random_integers(-(1<<63), (1<<63)-1, size=[size,size]), dtype='int64')
    A=tf.random.uniform(shape=[col_num], minval=-(1<<63), maxval=(1<<63)-1, dtype='int64')

    plaintext_out=A * plaintext

    cipher_v_out = homo_module.vec_mul_vec(pk, A, cipher)
    print("cipher_v_out=", cipher_v_out)



    plaintext_out2 = homo_module.dec(sk, [col_num], cipher_v_out)
    print(plaintext_out2)

    sess=tf.compat.v1.Session()

    sess.run(tf.compat.v1.initialize_all_variables())

    print(np.sum((sess.run(plaintext_out - plaintext_out2))**2))
    #sess=tf.compat.v1.Session()
    #print(sess.run(keys))





def test_cipher_mul_share():
    sk, pk, gk = homo_module.gen_key(tf.Variable(initial_value=[0])) # 一定要以这个tf.Variable作为入参，否则会有错误
    col_num = 100
    plaintext = random_module.rint64(shape=[col_num], step=tf.Variable(initial_value=[0]))
    #plaintext = tf.cast(tf.range(start=0, limit=size), 'int64') # (shape=[size], dtype='int64')

    cipher = homo_module.enc(pk, plaintext)
    print(cipher)

    cipher_v_out, share_v_out = homo_module.cipher_to_share(col_num, pk, cipher)
    print("cipher_v_out=", cipher_v_out)
    print("share_v_out=", share_v_out)

    share_v_out2 = homo_module.dec(sk, col_num, cipher_v_out)
    sess=tf.compat.v1.Session()

    sess.run(tf.compat.v1.initialize_all_variables())
    #print(sess.run([cipher_v_out, share_v_out]))
    print(np.sum((sess.run(share_v_out + share_v_out2 - plaintext))**2))



if __name__ == '__main__':
    test_mat_mul_vec()
    # test_vec_mul_vec()
    # test_cipher_mul_share()
    # test_mat_mul_vec_to_share()