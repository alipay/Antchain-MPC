from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, PrivateTensorBase
from stensorflow.basic.operator.conv2dop import *
import tensorflow as tf
import stensorflow as stf
import os
from stensorflow.engine.start_server import start_local_server
from stensorflow.random.random import random_init
from stensorflow.global_var import StfConfig
start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

# 初始化类函数

def back_filter_PP_test(input, out_back):
    """
    input和out_back都是Private，并且owner相同
    """
    # 转成PrivateTensor
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    p_out_back = stf.PrivateTensor(owner='L')
    p_out_back.load_from_tf_tensor(out_back)
    # 调用函数
    result = conv2d_backprop_filter(p_input, (2, 2, 1, 1), p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_filter_PPD_test(input, out_back):
    """
        input和out_back都是Private，但是owner不同
    """
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    p_out_back = stf.PrivateTensor(owner='R')
    p_out_back.load_from_tf_tensor(out_back)
    result = conv2d_backprop_filter(p_input, (2, 2, 1, 1), p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_filter_PS_test(input, out_back):
    """
        input是Private,out_back是Shared
    """
    p_input = stf.PrivateTensor(owner='L')
    p_input.load_from_tf_tensor(input)
    s_out_back = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 3, 3, 1])
    s_out_back.load_from_tf_tensor(out_back)
    result = conv2d_backprop_filter(p_input, (2, 2, 1, 1), s_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_filter_SP_test(input, out_back):
    """
        input是Shared,out_back是Private
    """
    s_input = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 3, 3, 1])
    s_input.load_from_tf_tensor(input)
    p_out_back = stf.PrivateTensor(owner='R')
    p_out_back.load_from_tf_tensor(out_back)
    result = conv2d_backprop_filter(s_input, (2, 2, 1, 1), p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_filter_SS_test(input, out_back):
    """
        input和out_back都是Shared
    """
    s_input = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 3, 3, 1])
    s_input.load_from_tf_tensor(input)
    s_out_back = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 3, 3, 1])
    s_out_back.load_from_tf_tensor(out_back)
    result = conv2d_backprop_filter(s_input, (2, 2, 1, 1), s_out_back, [1, 1, 1, 1], 'SAME')
    return result






if __name__ == "__main__":
    # 张量
    x = tf.constant(
        [
            [
                [[1], [2], [3]],
                [[4], [5], [6]],
                [[7], [8], [9]]
            ]
        ]
        , tf.float32
    )

    # 某一函数针对sigma的导数
    partial_sigma = tf.constant(
        [
            [
                [[-1], [-2], [1]],
                [[-3], [-4], [2]],
                [[-2], [1], [3]]
            ]
        ]
        , tf.float32
    )
    # 针对未知变量的导数的方向计算
    partial_sigma_k = tf.compat.v1.nn.conv2d_backprop_filter(x, (2, 2, 1, 1), partial_sigma, [1, 1, 1, 1], 'SAME')
    # 转化数据类型
    x = tf.cast(x, dtype=tf.int64)
    partial_sigma = tf.cast(partial_sigma, dtype=tf.int64)
    partial_sigma_k = tf.cast(partial_sigma_k, dtype=tf.int64)
    # result = back_filter_PP_test(x,partial_sigma)
    # result = back_filter_PPD_test(x, partial_sigma)
    # result = back_filter_PS_test(x, partial_sigma)
    # result = back_filter_SP_test(x, partial_sigma)
    result = back_filter_SS_test(x, partial_sigma)
    with tf.compat.v1.Session(StfConfig.target) as sess:
        print(result)
        random_init(sess)
        result = tf.cast(result.to_tf_tensor("R"), dtype=tf.int64)
        print(sess.run([partial_sigma_k, result]))
        print(sess.run(tf.equal(result, partial_sigma_k)))
