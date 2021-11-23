import unittest
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, PrivateTensorBase
from stensorflow.basic.operator.conv2dop import *
import tensorflow as tf
import stensorflow as stf
import os
from stensorflow.engine.start_server import start_local_server
from stensorflow.random.random import random_init
from stensorflow.global_var import StfConfig
import numpy as np
start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))


def PP_conv2d_test(input, filter):
    """
    input 和filter都是Private,并且owner相同
    :return:
    """
    # 转成PrivateTensor
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    p_filter = stf.PrivateTensor(owner="L")
    p_filter.load_from_tf_tensor(filter)
    # 调用函数
    result = conv2d(p_input, p_filter, strides=[1, 1, 1, 1], padding='SAME')
    return result


def PP_D_test(input, filter):
    """
    input 和filter都是Private,并且owner不同
    :return:
    """
    # 转成PrivateTensor
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    p_filter = stf.PrivateTensor(owner="R")
    p_filter.load_from_tf_tensor(filter)
    # 调用函数
    result = conv2d(p_input, p_filter, strides=[1, 1, 1, 1], padding='SAME')
    # 测试实验结果是否一致
    return result


def PS_test(input, filter):
    """
    input 是Private,filter是Shared
    :return:
    """
    # 转成PrivateTensor
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    # 保证shape和原始的shape一致才可
    s_filter = stf.SharedPair(ownerL="L", ownerR="R", shape=[5, 5, 1, 32])
    s_filter.load_from_tf_tensor(filter)
    # 调用函数
    result = conv2d(p_input, s_filter, strides=[1, 1, 1, 1], padding='SAME')
    # 测试实验结果是否一致
    return result


def SP_test(input, filter):
    """
    input是Shared，filter是Private
    :return:
    """

    # input转化为shared
    s_input = stf.SharedPair(ownerL="L",ownerR="R", shape=[1, 28, 28, 1])
    s_input.load_from_tf_tensor(input)
    # filter转为Private
    p_filter = stf.PrivateTensor(owner="L")
    p_filter.load_from_tf_tensor(filter)
    # 调用函数
    result = conv2d(s_input, p_filter, strides=[1, 1, 1, 1], padding='SAME')
    # 测试实验结果
    return result


def SS_test(input, filter):
    """
    input 和filter都是Shared类型
    :return:
    """
    # input转化为shared
    s_input = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 28, 28, 1])
    s_input.load_from_tf_tensor(input)
    # filter转为shared
    s_filter = stf.SharedPair(ownerL="L", ownerR="R", shape=[5, 5, 1, 32])
    s_filter.load_from_tf_tensor(filter)
    # 调用函数
    result = conv2d(s_input, s_filter, strides=[1, 1, 1, 1], padding='SAME')
    # 测试实验结果
    return result


if __name__ == "__main__":
    input = tf.constant(value=[i for i in range(28 * 28)], shape=[1, 28, 28, 1], dtype=tf.float64)
    filter = tf.constant(value=[i % 11 for i in range(5 * 5 * 32)], shape=[5, 5, 1, 32], dtype=tf.float64)
    # 标准操作结果
    r = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # 转化为int64
    input = tf.cast(input, dtype=tf.int64)
    filter = tf.cast(filter, dtype=tf.int64)
    r = tf.cast(r, dtype=tf.int64)
    print(r)
    # result = PP_conv2d_test(input, filter)
    # result = PP_D_test(input,filter)
    # result = PS_test(input, filter)
    # result = SP_test(input,filter)
    result = SS_test(input, filter)
    with tf.compat.v1.Session(StfConfig.target) as sess:
        print(result)
        random_init(sess)
        result = tf.cast(result.to_tf_tensor("R"), dtype=tf.int64)
        # print(sess.run([result.to_tf_tensor("R"), r]))
        print(sess.run(tf.equal(result, r)))


