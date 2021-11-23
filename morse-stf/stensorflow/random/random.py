#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : random.py
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/5/24 下午8:34
   Description : description what the main function of this file
"""
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.exception.exception import StfTypeException, StfCondException, check_is_not_None


def gen_rint64(shape):
    random_mod = StfConfig.random_module
    check_is_not_None(shape=shape)
    step = tf.Variable(initial_value=[0], trainable=False)
    tf.compat.v1.add_to_collection(StfConfig.coll_name_vars_random, step)
    x = random_mod.rint64(shape=shape, step=step)
    return x


def gen_ZZn(shape, module):
    random_mod = StfConfig.random_module
    check_is_not_None(shape=shape)
    step = tf.Variable(initial_value=[0], trainable=False)
    tf.compat.v1.add_to_collection(StfConfig.coll_name_vars_random, step)
    x = random_mod.rint64(shape=shape, step=step)
    return x % module


def get_seed():
    random_mod = StfConfig.random_module
    seed = random_mod.get_seed(0)
    return seed


def gen_rint64_from_seed(shape, seed):
    random_mod = StfConfig.random_module
    check_is_not_None(seed=seed)
    step = tf.Variable(initial_value=[0], dtype='int64', trainable=False)
    tf.compat.v1.add_to_collection(StfConfig.coll_name_vars_random, step)
    x = random_mod.rint64_from_seed(shape=shape, seed=seed + step)  # Only use seed in first time in this OP
    return x


def gen_ZZn_from_seed(shape, seed, module: int):
    if not isinstance(module, int):
        raise StfTypeException("module", 'int', type(module))
    elif module <= 0:
        raise StfCondException("module>0", "module={}".format(module))
    random_mod = StfConfig.random_module
    check_is_not_None(seed=seed)
    step = tf.Variable(initial_value=[0], dtype='int64', trainable=False)
    tf.compat.v1.add_to_collection(StfConfig.coll_name_vars_random, step)
    x = random_mod.rint64_from_seed(shape=shape, seed=seed + step) % module  # Only use seed in first time in this OP
    return x


def random_init(sess=None):
    init_coll = tf.compat.v1.get_collection(StfConfig.coll_name_vars_random)
    init_random_op = tf.compat.v1.initialize_variables(init_coll)
    if sess is not None:
        sess.run(init_random_op)
    else:
        return init_random_op
