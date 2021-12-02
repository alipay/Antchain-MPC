#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group Copyright (c) 2004-2021 All Rights Reserved.
"""
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from typing import Union
from stensorflow.basic.protocol.bilinear_triangle import BiliinearTriangle
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_SharedPair, BM_PrivateTensor_PrivateTensor, \
    BM_SharedPair_SharedPair, BM_SharedPair_PrivateTensor

model = StfConfig.conv_module


def conv2d(input: Union[PrivateTensorBase, SharedPairBase],
           filter: Union[PrivateTensorBase, SharedPairBase],
           strides,
           padding,
           data_format='NHWC',
           dilations=None,
           fixed_point=None,
           prf_flag=None) -> \
        Union[PrivateTensorBase, SharedPairBase]:
    """
    See tf.conv2d
    """
    if fixed_point is None:
        fixed_point = max(input.fixedpoint, filter.fixedpoint)
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    input_is_private = isinstance(input, PrivateTensorBase)
    input_is_pair = isinstance(input, SharedPairBase)
    filters_is_private = isinstance(filter, PrivateTensorBase)
    filters_is_pair = isinstance(filter, SharedPairBase)

    def func_conv2d(x, y):
        return model.int64_conv2d(input=x, filter=y, strides=strides, padding=padding,
                                  data_format=data_format, dilations=dilations)

    def lambda_func(x, y):
        return SharedTensorBase(func_conv2d(x.inner_value, y.inner_value))

    if input_is_private and filters_is_private:
        if input.owner == filter.owner:
            result = func_conv2d(x=input.inner_value, y=filter.inner_value)
            result = PrivateTensorBase(inner_value=result, fixedpoint=input.fixedpoint + filter.fixedpoint,
                                       owner=input.owner)
        else:
            result = BM_PrivateTensor_PrivateTensor(input, filter, lambda_func, prf_flag=prf_flag)
    elif input_is_private and filters_is_pair:
        result = BM_PrivateTensor_SharedPair(input, filter, lambda_func, prf_flag=prf_flag)
    elif input_is_pair and filters_is_private:
        result = BM_SharedPair_PrivateTensor(input, filter, lambda_func, prf_flag=prf_flag)
    elif input_is_pair and filters_is_pair:
        result = BM_SharedPair_SharedPair(input, filter, lambda_func, prf_flag=prf_flag)
    else:
        raise Exception("type exception for type(x)={}, type(y)={}".format(type(input), type(filter)))
    return result.dup_with_precision(fixed_point)


def conv2d_backprop_input(input_sizes,
                          filter: Union[PrivateTensorBase, SharedPairBase],
                          out_backprop: Union[PrivateTensorBase, SharedPairBase],
                          strides=None,
                          padding=None,
                          data_format="NHWC",
                          dilations=[1, 1, 1, 1],
                          fixed_point=None,
                          prf_flag=None
                          ) -> Union[PrivateTensorBase, SharedPairBase]:
    """
    See  tf.conv2d_backprop_input
    """
    if fixed_point is None:
        fixed_point = max(filter.fixedpoint, out_backprop.fixedpoint)
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    # whether an object is an instance of a class
    filters_is_private = isinstance(filter, PrivateTensorBase)
    filters_is_pair = isinstance(filter, SharedPairBase)
    out_backprop_is_private = isinstance(out_backprop, PrivateTensorBase)
    out_backprop_is_pair = isinstance(out_backprop, SharedPairBase)

    def func_back_input(f, o):
        return model.int64_conv2d_backprop_input(input_sizes=input_sizes, filter=f,
                                                 out_backprop=o, strides=strides,
                                                 padding=padding, data_format=data_format,
                                                 dilations=dilations)

    def lambda_func(f, o):
        return SharedTensorBase(func_back_input(f.inner_value, o.inner_value))

    if filters_is_private and out_backprop_is_private:
        if filter.owner == out_backprop.owner:
            result = func_back_input(filter.inner_value, out_backprop.inner_value)
            result = PrivateTensorBase(inner_value=result, fixedpoint=filter.fixedpoint + out_backprop.fixedpoint,
                                       owner=filter.owner)
        else:
            result = BM_PrivateTensor_PrivateTensor(filter, out_backprop, lambda_func, prf_flag=prf_flag)
    elif filters_is_private and out_backprop_is_pair:
        result = BM_PrivateTensor_SharedPair(filter, out_backprop, lambda_func, prf_flag=prf_flag)
    elif filters_is_pair and out_backprop_is_private:
        result = BM_SharedPair_PrivateTensor(filter, out_backprop, lambda_func, prf_flag=prf_flag)
    elif filters_is_pair and out_backprop_is_pair:
        result = BM_SharedPair_SharedPair(filter, out_backprop, lambda_func, prf_flag=prf_flag)
    else:
        raise Exception("type exception for type(x)={}, type(y)={}".format(type(filter), type(out_backprop)))
    return result.dup_with_precision(fixed_point)


def conv2d_backprop_filter(input: Union[PrivateTensorBase, SharedPairBase],
                           filter_sizes,
                           out_backprop: Union[PrivateTensorBase, SharedPairBase],
                           strides,
                           padding,
                           data_format="NHWC",
                           dilations=[1, 1, 1, 1],
                           fixed_point=None,
                           prf_flag=None
                           ) -> Union[PrivateTensorBase, SharedPairBase]:
    """
    See conv2d_backprop_filter
    """
    if fixed_point is None:
        fixed_point = max(input.fixedpoint, out_backprop.fixedpoint)
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    # whether an object is an instance of a class
    input_is_private = isinstance(input, PrivateTensorBase)
    input_is_pair = isinstance(input, SharedPairBase)
    out_backprop_is_private = isinstance(out_backprop, PrivateTensorBase)
    out_backprop_is_pair = isinstance(out_backprop, SharedPairBase)

    def func_back_filter(x, o):
        return model.int64_conv2d_backprop_filter(input=x,
                                                  filter_sizes=filter_sizes,
                                                  out_backprop=o,
                                                  strides=strides,
                                                  padding=padding,
                                                  data_format=data_format,
                                                  dilations=dilations)

    def lambda_func(x, o):
        return SharedTensorBase(func_back_filter(x.inner_value, o.inner_value))

    if input_is_private and out_backprop_is_private:
        if input.owner == out_backprop.owner:
            result = func_back_filter(input.inner_value, out_backprop.inner_value)
            result = PrivateTensorBase(inner_value=result,
                                       fixedpoint=input.fixedpoint + out_backprop.fixedpoint,
                                       owner=input.owner)
        else:
            result = BM_PrivateTensor_PrivateTensor(input, out_backprop, lambda_func, prf_flag=prf_flag)
    elif input_is_private and out_backprop_is_pair:
        result = BM_PrivateTensor_SharedPair(input, out_backprop, lambda_func, prf_flag=prf_flag)
    elif input_is_pair and out_backprop_is_private:
        result = BM_SharedPair_PrivateTensor(input, out_backprop, lambda_func, prf_flag=prf_flag)
    elif input_is_pair and out_backprop_is_pair:
        result = BM_SharedPair_SharedPair(input, out_backprop, lambda_func, prf_flag=prf_flag)
    else:
        raise Exception("type exception for type(x)={}, type(y)={}".format(type(input), type(out_backprop)))
    return result.dup_with_precision(fixed_point)


class Conv2dTriangle(BiliinearTriangle):
    def __init__(self, input_sizes, filter_sizes,
                   strides, padding,
                   data_format, dilations):
        def func_conv2d(x, y):
            return model.int64_conv2d(input=x, filter=y, strides=strides, padding=padding,
                                      data_format=data_format, dilations=dilations)

        def f_xy(x, y):
            return SharedTensorBase(func_conv2d(x.inner_value, y.inner_value))

        def func_back_input(f, ploss_pout):
            return model.int64_conv2d_backprop_input(input_sizes=input_sizes, filter=f,
                                                     out_backprop=ploss_pout, strides=strides,
                                                     padding=padding, data_format=data_format,
                                                     dilations=dilations)

        def f_yz(f, ploss_pout):
            return SharedTensorBase(func_back_input(f.inner_value, ploss_pout.inner_value))

        def func_back_filter(x, ploss_pout):
            return model.int64_conv2d_backprop_filter(input=x,
                                                      filter_sizes=filter_sizes,
                                                      out_backprop=ploss_pout,
                                                      strides=strides,
                                                      padding=padding,
                                                      data_format=data_format,
                                                      dilations=dilations)

        def f_zx(ploss_pout, x):
            return SharedTensorBase(func_back_filter(x.inner_value, ploss_pout.inner_value))

        super(Conv2dTriangle, self).__init__(f_xy, f_yz, f_zx)