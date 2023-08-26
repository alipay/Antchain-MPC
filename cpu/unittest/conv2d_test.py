#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
# 加载动态库
model = tf.load_op_library('/Users/jiaofuzhang/morse-stf/cops/_stf_int64conv2D_macos.so')
p_model = tf.load_op_library('/Users/jiaofuzhang/morse-stf/cops/_stf_int64pooling_macos.so')
# ST_model = tf.load_op_library('/Users/jiaofuzhang/morse-stf/cops/_std_pooling_macos.so')
from tensorflow.python.util.compat import collections_abc


def conv2d_forward():
    """
    测试conv2d前向计算过程
    :return:
    """
    # 生成int64 tensor
    input = tf.constant(value=[i for i in range(28 * 28)], shape=[1, 28, 28, 1], dtype=tf.int64)
    filter = tf.constant(value=[i % 11 for i in range(5 * 5 * 32)], shape=[5, 5, 1, 32], dtype=tf.int64)
    filter2 = tf.constant(value=[i % 17 for i in range(5 * 5 * 32 * 64)], shape=[5, 5, 32, 64], dtype=tf.int64)
    # 将数据转化为float类型测试数据结果
    f_input = tf.cast(input, dtype=tf.float64)
    f_filter = tf.cast(filter, dtype=tf.float64)
    f_filter2 = tf.cast(filter2, dtype=tf.float64)
    print(filter)
    # 测试自定义op
    op = model.int64_conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
    op = tf.nn.max_pool(op, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    op = model.int64_conv2d(op, filter2, strides=[1, 1, 1, 1], padding="SAME")
    op = tf.nn.max_pool(op, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # print(op)
    # 标准Conv2D操作
    fop = tf.nn.conv2d(f_input, f_filter, strides=[1, 1, 1, 1], padding='SAME')
    fop = tf.nn.max_pool(fop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    fop = tf.nn.conv2d(fop, f_filter2, strides=[1, 1, 1, 1], padding='SAME')
    fop = tf.nn.max_pool(fop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(fop)
    # 比较两个输出结果是否相同
    fop_c = tf.cast(fop, dtype=tf.int64)
    op = op.numpy()
    fop_c = fop_c.numpy()
    if (fop_c == op).all():
        print('conv2d forward success')
    else:
        print("conv2d forward fail")


def conv2d_backprop_input():
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
    # print(partial_x)
    # 转化为int64位的操作
    int64_kernel = tf.cast(kernel, dtype=tf.int64)
    int64_partial_sigma = tf.cast(partial_sigma, dtype=tf.int64)

    op_back_input = model.int64_conv2d_backprop_input(input_sizes=(1, 3, 3, 1),
                                                            filter=int64_kernel,
                                                            strides=[1, 1, 1, 1],
                                                            padding='SAME'
                                                            )
    # print(op_back_input)
    # 比较两个输出结果是否相同
    partial_x_to_INT64 = tf.cast(partial_x, dtype=tf.int64)
    # 转化为numpy
    op_back_input = op_back_input.numpy()
    partial_x_to_INT64 = partial_x_to_INT64.numpy()
    if (op_back_input == partial_x_to_INT64).all():
        print("Backprop Input success!")
    else:
        print("Backprop Input fail!")


def conv2d_backprop_filter():
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
    # print(partial_sigma_k)
    # 转化为INT64位操作
    INT64_x = tf.cast(x, dtype=tf.int64)
    INT64_partial_sigma = tf.cast(partial_sigma, dtype=tf.int64)
    # 加载动态库
    op_back_filter = model.int64_conv2d_backprop_filter(INT64_x, (2, 2, 1, 1), INT64_partial_sigma, [1, 1, 1, 1],
                                                               'SAME')
    # print(op_back_filter)
    # 比较结果是否一样
    partial_sigma_k_Int64 = tf.cast(partial_sigma_k, dtype=tf.int64)
    # 转化为Numpy
    partial_sigma_k_Int64 = partial_sigma_k_Int64.numpy()
    op_back_filter = op_back_filter.numpy()
    if (op_back_filter == partial_sigma_k_Int64).all():
        print("Backprop filter success!")
    else:
        print("Backprop filtert fail!")


def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  if value is None:
    value = [1]
  elif not isinstance(value, collections_abc.Sized):
    value = [value]

  current_n = len(value)
  if current_n == n + 2:
    return value
  elif current_n == 1:
    value = list((value[0],) * n)
  elif current_n == n:
    value = list(value)
  else:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, current_n))

  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]


def average_pooling_op():
    # 生成int64 tensor
    input = tf.constant(value=[-i for i in range(4 * 4)], shape=[1, 4, 4, 1], dtype=tf.int64)
    # print(input)
    # 将数据转化为float类型测试数据结果
    f_input = tf.cast(input, dtype=tf.float64)
    print(f_input.shape)
    channel_index = 3
    ksize = _get_sequence(2, 2, channel_index, "ksize")
    strides = _get_sequence(2, 2, channel_index, "strides")
    # 测试标准结果
    out = tf.nn.avg_pool2d(f_input, ksize=2, strides=2, padding="VALID")
    print(f_input)
    # float_out = ST_model.float64_avg_pool(f_input, ksize=ksize, strides=strides, padding="VALID")
    # exit()
    conv_out = tf.nn.conv2d(f_input, filters=np.ones([2, 2, 1, 1])/2/2, strides=strides, padding="VALID")
    print("比较conv2d和avgpool")
    # print(out)
    # print(conv_out)
    # print(conv_out)

    # 测试自定义op
    n_out = p_model.int64_avg_pool(input, ksize=ksize, strides=strides, padding="VALID")
    print(n_out)
    print(conv_out)
    print("比较梯度")
    # 测试梯度

    n_grad = p_model.int64_avg_pool_grad(orig_input_shape=[1, 4, 4, 1], grad=n_out, ksize=ksize, strides=strides,
                                         padding="VALID",data_format = "NHWC")
    print(n_grad)
    conv_grad = tf.compat.v1.nn.conv2d_backprop_input(
        input_sizes=(1, 4, 4, 1),
        filter=np.ones([2, 2, 1, 1])/2/2,
        out_backprop=conv_out,
        strides=strides,
        padding="VALID"
    )
    print(tf.cast(conv_grad, dtype=tf.int64))
    # print(n_grad.shape)
    # print(n_grad)
    # print(out)
    # print(n_out)
    # exit()
    # 测试结果是否相同
    out = tf.cast(out, dtype=tf.int64)
    out = out.numpy()
    n_out = n_out.numpy()

    # print(out)
    # print(n_out)
    if (n_out == out).all():
        print("hi")
    else:
        print("cry")


def cal_sum_pooling():
    strides = [1, 1, 1, 1]
    ksize = [1, 2, 2, 1]
    x = tf.random.uniform(shape=[1, 3, 3, 1], minval=0, maxval=1 << 62, dtype='int64')
    print(x)
    y = p_model.int64_avg_pool(x, ksize=ksize, strides=strides, padding="VALID")
    print(y)
    exit()
    plosspy1 = tf.random.uniform(shape=[1, 2, 2, 1], minval=0, maxval=1 << 62, dtype='int64')
    plosspy2 = tf.random.uniform(shape=[1, 2, 2, 1], minval=0, maxval=1 << 62, dtype='int64')
    plosspy = plosspy2+plosspy1
    plosspx = p_model.sum_pool_grad(orig_input_shape=[1, 3, 3, 1], grad=plosspy,
                                    ksize=ksize, strides=strides, padding="VALID", data_format="NHWC")
    plosspx1 = p_model.sum_pool_grad(orig_input_shape=[1, 3, 3, 1], grad=plosspy1,
                                    ksize=ksize, strides=strides, padding="VALID", data_format="NHWC")
    plosspx2 = p_model.sum_pool_grad(orig_input_shape=[1, 3, 3, 1], grad=plosspy2,
                                     ksize=ksize, strides=strides, padding="VALID", data_format="NHWC")
    y = plosspx1+plosspx2

    print(plosspx)
    print(y)
    # print(plosspx/4)








if __name__ == "__main__":
    # conv2d_forward()
    # conv2d_backprop_input()
    # conv2d_backprop_filter()
    # average_pooling_op()
    cal_sum_pooling()




