#!/usr/bin/env python
# coding=utf-8
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.exception.exception import StfNoneException, StfCondException
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateVariable
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.operator.conv2dop import *
from tensorflow.python.keras.utils import conv_utils
from typing import Union, List
from stensorflow.basic.basic_class.base import get_device
from tensorflow.python.ops import random_ops
import math


def initializers(kernel_shape):
    """
    :param kernel_shape: [channel_in,height,weight,channel_out]
    :return: A Tensor
    """
    fan_in = kernel_shape[0]*kernel_shape[1]*kernel_shape[2]
    fan_out = kernel_shape[0]*kernel_shape[1]*kernel_shape[3]
    scale = 1  # default
    scale /= max(1., (fan_in + fan_out) / 2.)
    stddev = math.sqrt(scale) / .87962566103423978
    ans = random_ops.truncated_normal(kernel_shape, 0.0, stddev, seed=None)
    return ans


class Conv2d_bak(Layer):
    """
    Implement the conv2d layer of the secure computing version
    This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    If `use_bias` is True,a bias vector is created and added to the outputs.
    Arguments:
        output_dim: output_shape
        fathers: layer
        filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution),
        kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
        input_shape: A list
        strides:list of `ints` that has length  `4`.
          stride of the sliding window for each dimension of `input`.
          The dimension order is determined
          by the value of `data_format`, see below for details.
        padding:Either the `string` `"SAME"` or `"VALID"` indicating the type of
          padding algorithm to use, or a list indicating the explicit paddings at
          the start and end of each dimension. When explicit padding is used and
          data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
          pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
          and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
          [pad_top, pad_bottom], [pad_left, pad_right]]`.
        data_format:
        default format "NHWC", the data is stored in the order of:
              [batch, height, width, channels].
        dilations:
            An  list of `ints` that has length  `4`
          defaults to [1,1,1,1]. The dilation factor for each dimension of`input`. If a
          single value is given it is replicated in the `H` and `W` dimension. By
          default the `N` and `C` dimensions are set to 1. If set to k > 1, there
          will be k-1 skipped cells between each filter element on that dimension.
          The dimension order is determined by the value of `data_format`, see above
          for details. Dilations in the batch and depth dimensions if a 4-d tensor must be 1.
        prf_flag:
        use_bias: Boolean, whether the layer uses a bias.
    Return:
        Has the same type as `input`.

    """
    def __init__(self,
                 output_dim=None,
                 fathers=None,
                 filters=1,
                 kernel_size=3,
                 input_shape=None,
                 strides=[1, 1, 1, 1],
                 padding='VALID',
                 data_format="NHWC",
                 dilations=[1, 1, 1, 1],
                 prf_flag=None
                 ):
        if fathers is None:
            raise StfNoneException("fathers")
        if not fathers:
            raise StfCondException("fathers != []", "fathers == []")

        if filters is not None and not isinstance(filters, int):
            filters = int(filters)
        self.filters = filters

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')

        if not isinstance(input_shape, list):
            raise TypeError(
                "Expected list for 'input_shape' argument to "
                "'conv2d' Op, not %r." % input_shape)
        self.input_shape = input_shape
        if not isinstance(strides, (int, list, tuple)):
            raise TypeError(
                "Expected int or list for 'strides' argument to "
                "'conv2d' Op, not %r." % strides)
        self.strides = strides
        self.padding = padding
        if data_format is None:
            data_format = "NHWC"
        self.data_format = data_format
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if not isinstance(dilations, (list, tuple)):
            raise TypeError(
                "Expected list for 'dilations' argument to "
                "'conv2d' Op, not %r." % dilations)
        self.dilations = dilations
        if data_format=="NHWC":
            input_channel = self.input_shape[2]
        elif data_format =="NCHW":
            input_channel = self.input_shape[1]
        else:
            raise TypeError(
                "Expected 'NHWC' or NCHW argument to "
                "'data format', not %r." % self.data_format)

        if not output_dim:
            output_dim = self.compute_shape()
        super(Conv2d, self).__init__(output_dim, fathers)

        self.prf_flg = prf_flag

        self.kernel_shape = self.kernel_size + (input_channel, self.filters)
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")

            kernel = SharedVariablePair(ownerL="L", ownerR="R", shape=self.kernel_shape)
            kernel.load_from_tf_tensor(initializers(self.kernel_shape))
            self.w += [kernel]

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        """
        Forward propagation
        Arguments:
            w: filter, SharedVariablePair
            x: input , PrivateTensor or SharedPair
        Return:
             Has the same type as `input`.
        """

        y = conv2d(input=x[0], filter=w[0],
                   strides=self.strides, padding=self.padding,
                   data_format=self.data_format, dilations=self.dilations,
                   fixed_point=x[0].fixedpoint, prf_flag=self.prf_flg)
        self.back_shape = x[0].shape
        y = SharedPair(ownerL=y.ownerL, ownerR=y.ownerR, xL=y.xL, xR=y.xR, fixedpoint=y.fixedpoint)
        return y

    def pull_back(self,  w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]],
                  y: SharedPair, ploss_py: SharedPair):
        """
        Back propagation
        Arguments:
            w: filter
            x: input
            y: output
            ploss_py: delta
        :return:
        """
        # input backprop
        # print("conv2d back ")
        batch_size = x[0].shape[0]
        ploss_px = conv2d_backprop_input(input_sizes=x[0].shape, filter=w[0],
                                         out_backprop=ploss_py, strides=self.strides,
                                         padding=self.padding, data_format=self.data_format,
                                         dilations=self.dilations,
                                         fixed_point=x[0].fixedpoint,
                                         prf_flag=self.prf_flg)

        ploss_px = SharedPair(ownerL=ploss_px.ownerL, ownerR=ploss_px.ownerR,
                              xL=ploss_px.xL, xR=ploss_px.xR, fixedpoint=ploss_px.fixedpoint)
        # filter backprop
        ploss_pw = conv2d_backprop_filter(input=x[0], filter_sizes=self.kernel_shape,
                                          out_backprop=ploss_py, strides=self.strides,
                                          padding=self.padding, data_format=self.data_format,
                                          dilations=self.dilations, fixed_point=x[0].fixedpoint,
                                          prf_flag=self.prf_flg)

        if isinstance(ploss_pw, PrivateTensorBase):
            ploss_pw = PrivateTensor.from_PrivteTensorBase(ploss_pw)
        if isinstance(ploss_pw, SharedPairBase):
            ploss_pw = SharedPair(ownerL=ploss_pw.ownerL, ownerR=ploss_pw.ownerR,
                                  xL=ploss_pw.xL, xR=ploss_pw.xR, fixedpoint=ploss_pw.fixedpoint)

        ploss_pw = ploss_pw/batch_size
        ploss_px = {self.fathers[0]: ploss_px}
        return [ploss_pw], ploss_px

    def compute_shape(self):
        # return output shape
        new_space = []
        if self.data_format == "NHWC":
            space = self.input_shape[:-1]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i+1],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res
        else:
            space = self.input_shape[1:]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res



class Conv2d(Layer):
    """
    Implement the conv2d layer of the secure computing version
    This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    If `use_bias` is True,a bias vector is created and added to the outputs.
    Arguments:
        output_dim: output_shape
        fathers: layer
        filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution),
        kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
        input_shape: A list
        strides:list of `ints` that has length  `4`.
          stride of the sliding window for each dimension of `input`.
          The dimension order is determined
          by the value of `data_format`, see below for details.
        padding:Either the `string` `"SAME"` or `"VALID"` indicating the type of
          padding algorithm to use, or a list indicating the explicit paddings at
          the start and end of each dimension. When explicit padding is used and
          data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
          pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
          and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
          [pad_top, pad_bottom], [pad_left, pad_right]]`.
        data_format:
        default format "NHWC", the data is stored in the order of:
              [batch, height, width, channels].
        dilations:
            An  list of `ints` that has length  `4`
          defaults to [1,1,1,1]. The dilation factor for each dimension of`input`. If a
          single value is given it is replicated in the `H` and `W` dimension. By
          default the `N` and `C` dimensions are set to 1. If set to k > 1, there
          will be k-1 skipped cells between each filter element on that dimension.
          The dimension order is determined by the value of `data_format`, see above
          for details. Dilations in the batch and depth dimensions if a 4-d tensor must be 1.
        prf_flag:
        use_bias: Boolean, whether the layer uses a bias.
    Return:
        Has the same type as `input`.

    """
    def __init__(self,
                 output_dim=None,
                 fathers=None,
                 filters=1,
                 kernel_size=3,
                 input_shape=None,
                 strides=[1, 1, 1, 1],
                 padding='VALID',
                 data_format="NHWC",
                 dilations=[1, 1, 1, 1],
                 prf_flag=None
                 ):
        if fathers is None:
            raise StfNoneException("fathers")
        if not fathers:
            raise StfCondException("fathers != []", "fathers == []")

        if filters is not None and not isinstance(filters, int):
            filters = int(filters)
        self.filters = filters

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')

        if not isinstance(input_shape, list):
            raise TypeError(
                "Expected list for 'input_shape' argument to "
                "'conv2d' Op, not %r." % input_shape)
        self.input_shape = input_shape
        if not isinstance(strides, (int, list, tuple)):
            raise TypeError(
                "Expected int or list for 'strides' argument to "
                "'conv2d' Op, not %r." % strides)
        self.strides = strides
        self.padding = padding
        if data_format is None:
            data_format = "NHWC"
        self.data_format = data_format
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if not isinstance(dilations, (list, tuple)):
            raise TypeError(
                "Expected list for 'dilations' argument to "
                "'conv2d' Op, not %r." % dilations)
        self.dilations = dilations
        if data_format=="NHWC":
            input_channel = self.input_shape[2]
        elif data_format =="NCHW":
            input_channel = self.input_shape[1]
        else:
            raise TypeError(
                "Expected 'NHWC' or NCHW argument to "
                "'data format', not %r." % self.data_format)

        if not output_dim:
            output_dim = self.compute_shape()
        super(Conv2d, self).__init__(output_dim, fathers)

        self.prf_flg = prf_flag

        self.kernel_shape = self.kernel_size + (input_channel, self.filters)
        self.bt_list = []
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")

            kernel = SharedVariablePair(ownerL="L", ownerR="R", shape=self.kernel_shape)
            kernel.load_from_tf_tensor(initializers(self.kernel_shape))
            self.w += [kernel]


    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        """
        Forward propagation
        Arguments:
            w: filter, SharedVariablePair
            x: input , PrivateTensor or SharedPair
        Return:
             Has the same type as `input`.
        """

        # y = conv2d(input=x[0], filter=w[0],
        #            strides=self.strides, padding=self.padding,
        #            data_format=self.data_format, dilations=self.dilations,
        #            fixed_point=x[0].fixedpoint, prf_flag=self.prf_flg)
        # self.back_shape = x[0].shape
        #y = SharedPair(ownerL=y.ownerL, ownerR=y.ownerR, xL=y.xL, xR=y.xR, fixedpoint=y.fixedpoint)
        self.bt = Conv2dTriangle(input_sizes=x[0].shape, filter_sizes=w[0].shape,
                            strides=self.strides, padding=self.padding, data_format=self.data_format,
                            dilations=self.dilations)
        y = self.bt.compute_u(x[0], w[0])
        return y.dup_with_precision(x[0].fixedpoint)

    def pull_back(self,  w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]],
                  y: SharedPair, ploss_py: SharedPair):
        """
        Back propagation
        Arguments:
            w: filter
            x: input
            y: output
            ploss_py: delta
        :return:
        """
        # input backprop
        # print("conv2d back ")
        # ploss_px = conv2d_backprop_input(input_sizes=x[0].shape, filter=w[0],
        #                                  out_backprop=ploss_py, strides=self.strides,
        #                                  padding=self.padding, data_format=self.data_format,
        #                                  dilations=self.dilations,
        #                                  fixed_point=x[0].fixedpoint,
        #                                  prf_flag=self.prf_flg)
        #
        # ploss_px = SharedPair(ownerL=ploss_px.ownerL, ownerR=ploss_px.ownerR,
        #                       xL=ploss_px.xL, xR=ploss_px.xR, fixedpoint=ploss_px.fixedpoint)
        # # filter backprop
        # ploss_pw = conv2d_backprop_filter(input=x[0], filter_sizes=self.kernel_shape,
        #                                   out_backprop=ploss_py, strides=self.strides,
        #                                   padding=self.padding, data_format=self.data_format,
        #                                   dilations=self.dilations, fixed_point=x[0].fixedpoint,
        #                                   prf_flag=self.prf_flg)
        ploss_px, ploss_pw = self.bt.compute_vw(ploss_py)

        if isinstance(ploss_pw, PrivateTensorBase):
            ploss_pw = PrivateTensor.from_PrivteTensorBase(ploss_pw)
        if isinstance(ploss_pw, SharedPairBase):
            ploss_pw = SharedPair(ownerL=ploss_pw.ownerL, ownerR=ploss_pw.ownerR,
                                  xL=ploss_pw.xL, xR=ploss_pw.xR, fixedpoint=ploss_pw.fixedpoint)
        batch_size = x[0].shape[0]
        ploss_pw = ploss_pw/batch_size
        ploss_px = {self.fathers[0]: ploss_px}
        return [ploss_pw], ploss_px

    def compute_shape(self):
        # return output shape
        new_space = []
        if self.data_format == "NHWC":
            space = self.input_shape[:-1]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i+1],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res
        else:
            space = self.input_shape[1:]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res





class Conv2dLocal(Layer):
    def __init__(self,
                 output_dim,
                 fathers,
                 owner,
                 filters=1,
                 kernel_size=3,
                 input_shape=None,
                 strides=[1, 1, 1, 1],
                 padding='VALID',
                 data_format="NHWC",
                 dilations=[1, 1, 1, 1],
                 prf_flag=None):
        if fathers is None:
            raise StfNoneException("fathers")
        if not fathers:
            raise StfCondException("fathers != []", "fathers == []")

        self.owner = get_device(owner)

        if filters is not None and not isinstance(filters, int):
            filters = int(filters)
        self.filters = filters

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')

        if not isinstance(input_shape, list):
            raise TypeError(
                "Expected list for 'input_shape' argument to "
                "'conv2d' Op, not %r." % input_shape)
        self.input_shape = input_shape
        if not isinstance(strides, (list, tuple)):
            raise TypeError(
                "Expected list for 'strides' argument to "
                "'conv2d' Op, not %r." % strides)
        self.strides = strides
        self.padding = padding
        if data_format is None:
            data_format = "NHWC"
        self.data_format = data_format
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if not isinstance(dilations, (list, tuple)):
            raise TypeError(
                "Expected list for 'dilations' argument to "
                "'conv2d' Op, not %r." % dilations)
        self.dilations = dilations
        if data_format=="NHWC":
            input_channel = self.input_shape[2]
        elif data_format =="NCHW":
            input_channel = self.input_shape[1]
        else:
            raise TypeError(
                "Expected 'NHWC' or NCHW argument to "
                "'data format', not %r." % self.data_format)

        if not output_dim:
            output_dim = self.compute_shape()
        super(Conv2dLocal, self).__init__(output_dim, fathers)
        self.prf_flg = prf_flag
        self.kernel_shape = self.kernel_size + (input_channel, self.filters)
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")

            kernel = PrivateVariable(owner=self.owner)
            kernel.load_from_tf_tensor(initializers(self.kernel_shape))
            self.w += [kernel]

    def func(self, w: List[PrivateTensor], x: List[Union[PrivateTensor]]):
        """
        :param w: filter，Private
        :param x: input,Private
        :return:
        """
        self.back_shape = x[0].shape

        y = conv2d(input=PrivateTensor.from_PrivteTensorBase(x[0].to_private(owner=self.owner), op_map=x[0].op_map),
                   filter=w[0],
                   strides=self.strides, padding=self.padding,
                   data_format=self.data_format, dilations=self.dilations,
                   fixed_point=x[0].fixedpoint, prf_flag=self.prf_flg)
        y = PrivateTensor.from_PrivteTensorBase(y)
        y = y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)
        return y

    def pull_back(self, w: List[PrivateTensor], x: List[Union[PrivateTensor, SharedPair]],
                  y: SharedPair, ploss_py: SharedPair):

        ploss_py = ploss_py.to_private(owner=self.owner)
        ploss_py = PrivateTensor.from_PrivteTensorBase(ploss_py)
        x = PrivateTensor.from_PrivteTensorBase(x[0].to_private(owner=self.owner))
        batch_size = x[0].shape[0]
        ploss_px = conv2d_backprop_input(input_sizes=x.shape, filter=w[0],
                                         out_backprop=ploss_py, strides=self.strides,
                                         padding=self.padding, data_format=self.data_format,
                                         dilations=self.dilations,
                                         fixed_point=x.fixedpoint,
                                         prf_flag=self.prf_flg)
        ploss_px = PrivateTensor.from_PrivteTensorBase(ploss_px)
        # filter backprop
        ploss_pw = conv2d_backprop_filter(input=x, filter_sizes=self.kernel_shape,
                                          out_backprop=ploss_py, strides=self.strides,
                                          padding=self.padding, data_format=self.data_format,
                                          dilations=self.dilations, fixed_point=x.fixedpoint,
                                          prf_flag=self.prf_flg)
        ploss_pw = PrivateTensor.from_PrivteTensorBase(ploss_pw)
        ploss_pw = ploss_pw/batch_size
        ploss_px = {self.fathers[0]: ploss_px}

        return [ploss_pw], ploss_px

    def compute_shape(self):
        # return output shape
        new_space = []
        if self.data_format == "NHWC":
            space = self.input_shape[:-1]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res
        else:
            space = self.input_shape[1:]
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding.lower(),
                    stride=self.strides[i],
                    dilation=self.dilations[i])
                new_space.append(new_dim)
            res = new_space + [self.filters]
            return res









