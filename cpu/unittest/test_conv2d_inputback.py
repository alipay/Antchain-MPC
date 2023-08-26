
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


def back_input_PP_test(kernel, out_back):
    """
    kernel 和 out_back都是Private，并且owener相同
    :param kernel:
    :param out_back:
    :return:
    """
    # 转成PrivateTensor
    p_kernel = stf.PrivateTensor(owner='L')
    p_kernel.load_from_tf_tensor(kernel)
    p_out_back = stf.PrivateTensor(owner='L')
    p_out_back.load_from_tf_tensor(out_back)
    # 调用函数
    result = conv2d_backprop_input((1, 3, 3, 1), p_kernel, p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_input_PPD_test(kernel, out_back):
    """
        kernel 和 out_back都是Private，但是owener不同
        """
    # 转成PrivateTensor
    p_kernel = stf.PrivateTensor(owner='L')
    p_kernel.load_from_tf_tensor(kernel)
    p_out_back = stf.PrivateTensor(owner='R')
    p_out_back.load_from_tf_tensor(out_back)
    # 调用函数
    result = conv2d_backprop_input((1, 3, 3, 1), p_kernel, p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_input_PS_test(kernel, out_back):
    """
    kernel是Private，out_back是Shared
    :param kernel:
    :param out_back:
    :return:
    """
    # 转成PrivateTensor
    p_kernel = stf.PrivateTensor(owner='L')
    p_kernel.load_from_tf_tensor(kernel)
    # 转成Shared
    s_out_back = stf.SharedPair(ownerL="L",ownerR="R", shape=[1, 3, 3, 1])
    s_out_back.load_from_tf_tensor(out_back)
    # 调用函数
    result = conv2d_backprop_input((1, 3, 3, 1), p_kernel, s_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_input_SP_test(kernel, out_back):
    """
        kernel是Shared，out_back是Private
    """
    s_kernel = stf.SharedPair(ownerL="L", ownerR="R", shape=[2, 2, 1, 1])
    s_kernel.load_from_tf_tensor(kernel)
    p_out_back = stf.PrivateTensor(owner='L')
    p_out_back.load_from_tf_tensor(out_back)
    # 调用函数
    result = conv2d_backprop_input((1, 3, 3, 1), s_kernel, p_out_back, [1, 1, 1, 1], 'SAME')
    return result


def back_input_SS_test(kernel, out_back):
    """
    kernel 和 out_back都是Shared

    """
    s_kernel = stf.SharedPair(ownerL="L", ownerR="R", shape=[2, 2, 1, 1])
    s_kernel.load_from_tf_tensor(kernel)
    s_out_back = stf.SharedPair(ownerL="L", ownerR="R", shape=[1, 3, 3, 1])
    s_out_back.load_from_tf_tensor(out_back)
    result = conv2d_backprop_input((1, 3, 3, 1), s_kernel, s_out_back, [1, 1, 1, 1], 'SAME')
    return result




if __name__ == "__main__":
    # 卷积核
    kernel = tf.constant(
        [
            [[[3]], [[4]]],
            [[[5]], [[6]]]
        ]
        , tf.float32
    )
    # 某一函数针对sigma的导数
    partial_sigma = tf.constant(
        [
            [
                [[-1], [1], [3]],
                [[2], [-2], [-4]],
                [[-3], [4], [1]]
            ]
        ]
        , tf.float32
    )
    # 针对未知变量的导数的方向计算
    partial_x = tf.compat.v1.nn.conv2d_backprop_input((1, 3, 3, 1), kernel, partial_sigma, [1, 1, 1, 1], 'SAME')
    # 转化为int64位的操作
    kernel = tf.cast(kernel, dtype=tf.int64)
    partial_sigma = tf.cast(partial_sigma, dtype=tf.int64)
    partial_x = tf.cast(partial_x, dtype=tf.int64)
    # 测试函数效果
    # result = back_input_PP_test(kernel, partial_sigma)
    # result = back_input_PPD_test(kernel,partial_sigma)
    # result = back_input_PS_test(kernel, partial_sigma)
    # result = back_input_SP_test(kernel, partial_sigma)
    result = back_input_SS_test(kernel, partial_sigma)
    with tf.compat.v1.Session(StfConfig.target) as sess:
        print(result)
        random_init(sess)
        result = tf.cast(result.to_tf_tensor("R"), dtype=tf.int64)
        print(sess.run([partial_x, result]))
        print(sess.run(tf.equal(result, partial_x)))
