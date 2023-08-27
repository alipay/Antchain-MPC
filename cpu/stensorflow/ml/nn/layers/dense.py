#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : Dense
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-09-11 15:57
   Description : description what the main function of this file
"""

from stensorflow.ml.nn.layers.layer import Layer
import numpy as np
from stensorflow.global_var import StfConfig
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateVariable
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.basic_class.base import get_device
from typing import Union, List
from stensorflow.basic.operator.relu import drelu_binary, relu, relu_pull_back, drelu_local
from stensorflow.basic.operator.truncation import dup_with_precision
from stensorflow.basic.protocol.bilinear_triangle import BilinearTriangle
import pandas as pd
from stensorflow.exception.exception import StfNoneException, StfCondException
import tensorflow as tf
#
# class Dense(Layer):
#     """
#     Dense Layer
#     """
#     def __init__(self, output_dim, fathers, with_b=True):
#         if fathers is None:
#             raise StfNoneException("fathers")
#         if fathers == []:
#             raise StfCondException("fathers != []", "fathers == []")
#         super(Dense, self).__init__(output_dim=output_dim, fathers=fathers)
#         self.with_b = with_b
#         for father in fathers:
#             if not isinstance(father, Layer):
#                 raise Exception("father must be a layer")
#             wi = SharedVariablePair(ownerL="L", ownerR="R", shape=[father.output_dim, output_dim])
#             wi.load_from_numpy(
#                 np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))
#             self.w += [wi]
#
#         if with_b:
#             b = SharedVariablePair(ownerL="L", ownerR="R", shape=[output_dim])
#             b.load_from_numpy(np.zeros([output_dim]))
#             self.w += [b]
#
#     def __str__(self):
#         return "Dense Layer of output_dim={}".format(self.output_dim)
#
#     def __repr__(self):
#         return self.__str__()
#
#     def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
#         if len(w) != len(x) + 1:
#             raise Exception("must have len(w)==len(x)+1")
#
#         y = x[0] @ w[0]
#         y = y.dup_with_precision(x[0].fixedpoint)
#         for i in range(1, len(x)):
#             y = y + x[i] @ w[i]
#         if self.with_b:
#             y = y + self.w[len(x)]
#         return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)
#
#     def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
#                   ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
#         batch_size = x[0].shape[0]
#         list_ploss_px = []
#         ploss_pw = []
#         for i in range(len(x)):
#             ploss_pxi = ploss_py @ w[i].transpose()
#             list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]
#
#             ploss_pwi = x[i].transpose() @ ploss_py
#             ploss_pwi = ploss_pwi.dup_with_precision(x[0].fixedpoint)
#             ploss_pwi = ploss_pwi / batch_size
#             ploss_pw += [ploss_pwi.dup_with_precision(x[0].fixedpoint)]
#
#         ploss_px = dict(zip(self.fathers, list_ploss_px))
#
#         if self.with_b:
#             ploss_pb = ploss_py.reduce_sum(axis=[0]) / batch_size
#             ploss_pw += [ploss_pb.dup_with_precision(x[0].fixedpoint)]
#
#         return ploss_pw, ploss_px
#
#     def save(self, save_file_machine, sess, path):
#         j = 0
#         for weight in self.w:
#             weight = weight.to_tf_tensor(owner=save_file_machine)
#             weight = sess.run(weight)
#             weight = pd.DataFrame(data=weight)
#             weight.to_csv(path + "_{}".format(j), header=False, index=False)
#             j += 1
#
#     def load(self, path):
#         j = 0
#         w = []
#         for weight in self.w:
#             assert isinstance(weight, SharedVariablePair) or isinstance(weight, PrivateVariable)
#             value = pd.read_csv(path + "_{}".format(j), header=None, index_col=None)
#             value = np.array(value)
#             value = np.reshape(value, weight.shape)
#             weight.load_from_numpy(value, const=True)
#             w += [weight]
#             j += 1
#         self.w = w

forward_call_id = 0
backward_call_id = 0
class Dense(Layer):
    """
    Dense Layer
    """
    def __init__(self, output_dim, fathers, with_b=True, activate=None):
        if fathers is None:
            raise StfNoneException("fathers")
        if fathers == []:
            raise StfCondException("fathers != []", "fathers == []")
        super(Dense, self).__init__(output_dim=output_dim, fathers=fathers)
        self.with_b = with_b
        self.bt_list = []
        self.activate = activate
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")
            wi = SharedVariablePair(ownerL="L", ownerR="R", shape=[father.output_dim, output_dim])
            wi.load_from_numpy(
                np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))
            self.w += [wi]
            f_xy = lambda a, b: a@b
            f_yz = lambda b, c: b@c.transpose()
            f_zx = lambda c, a: c.transpose()@a
            bt = BilinearTriangle(f_xy, f_yz, f_zx) # x, w, plosspy^T
            self.bt_list.append(bt)

        if with_b:
            b = SharedVariablePair(ownerL="L", ownerR="R", shape=[output_dim])
            b.load_from_numpy(np.zeros([output_dim], dtype='int64'))
            self.w += [b]

    def __str__(self):
        return "Dense Layer of output_dim={},".format(self.output_dim)+ "fathers output_dim=" + ", ".join("{}".format(father.output_dim) for father in self.fathers)

    def __repr__(self):
        return self.__str__()


    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        global forward_call_id
        if len(w) != len(x) + 1:
            raise Exception("must have len(w)==len(x)+1")

        # y = x[0] @ w[0]
        # print_obj = ("l160, forward_call_id=", forward_call_id, "\n x[0]= \n", tf.norm(x[0].to_tf_tensor("R")))

        # print_obj += ("\n l162, forward_call_id=", forward_call_id, "\n w[0]= \n", tf.norm(w[0].to_tf_tensor("R")))
        #StfConfig.log_op_list.append(print_op)

        y = self.bt_list[0].compute_u(x[0], w[0])
        # print_obj += ("\n l166, forward_call_id=", forward_call_id, "\n y= \n", tf.norm(y.to_tf_tensor("R")))
        for i in range(1, len(x)):
            # y = y + x[i] @ w[i]
            y = y + self.bt_list[i].compute_u(x[i], w[i])
        # print_obj += ("\n l169, forward_call_id=", forward_call_id, "\n y= \n", tf.norm(y.to_tf_tensor("R")))
        #StfConfig.log_op_list.append(print_op)
        if self.with_b:
            # print_obj += ("\n l166, forward_call_id=", forward_call_id, "\n b= \n", tf.norm(self.w[len(x)].to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            b = self.w[len(x)].dup_with_precision(new_fixedpoint=y.fixedpoint, non_negative=False)
            # print_obj += ("\n l170, forward_call_id=", forward_call_id, "\n b= \n", tf.norm(b.to_tf_tensor("R")) )
            y = y + b
            # print_obj += ("\n l176, forward_call_id=", forward_call_id, "\n y= \n", tf.norm(y.to_tf_tensor("R")))
        if self.activate is not None:
            if self.activate == 'relu':
                self.s = drelu_binary(y)
                out = relu(y, self.s)
                # print_obj += ("\n l180, forward_call_id=", forward_call_id, "\n out= \n", tf.norm(out.to_tf_tensor("R")))
                if StfConfig.positive_truncation_without_error:
                    out = out.dup_with_precision(x[0].fixedpoint, non_negative=True)
                else:
                    out = out.dup_with_precision(x[0].fixedpoint)
            else:
                raise Exception("unsupported activate function")
        else:
            out = y.dup_with_precision(x[0].fixedpoint, non_negative=False)
        # print_obj += ("\n l177, forward_call_id=", forward_call_id, "\n out= \n", tf.norm(out.to_tf_tensor("R")))
        # print_op = tf.print(*# print_obj)
        # StfConfig.log_op_list.append(print_op)
        forward_call_id +=1
        return out


    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
        global backward_call_id
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []
        # print_obj = ()
        if self.activate is not None:
            if self.activate == 'relu':
                ploss_py = relu_pull_back(None, ploss_py, drelu_b=self.s)
                # print_obj += ("l201, backward_call_id=", backward_call_id, "\n ploss_py= \n", tf.norm(ploss_py.to_tf_tensor("R")))
                # StfConfig.log_op_list.append(print_op)
            else:
                raise Exception("unsupported activate function")
        for i in range(len(x)):
            ploss_pxi_t, ploss_pwi_t = self.bt_list[i].compute_vw(ploss_py)
            # print_obj += ("\n l207, backward_call_id=", backward_call_id, "\n ploss_pxi_t= \n", tf.norm(ploss_pxi_t.to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            # print_obj += ("\n l207, backward_call_id=", backward_call_id, "\n ploss_pwi_t= \n", tf.norm(ploss_pwi_t.to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            ploss_pxi = ploss_pxi_t.transpose()
            ploss_pwi = ploss_pwi_t.transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]
            ploss_pwi = ploss_pwi.dup_with_precision(x[0].fixedpoint)
            ploss_pwi = ploss_pwi / batch_size
            # print_obj += ("\n l216, backward_call_id=", backward_call_id, "\n ploss_pwi= \n", tf.norm(ploss_pwi.to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            ploss_pw += [ploss_pwi.dup_with_precision(x[0].fixedpoint)]

        ploss_px = dict(zip(self.fathers, list_ploss_px))

        if self.with_b:
            ploss_pb = ploss_py.reduce_sum(axis=[0])
            # print_obj += ("\n l210, backward_call_id=", backward_call_id, "\n ploss_pb= \n", tf.norm(ploss_pb.to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            ploss_pb = ploss_pb / batch_size
            # print_obj += ("\n l215, backward_call_id=", backward_call_id, "\n ploss_pb= \n", tf.norm(ploss_pb.to_tf_tensor("R")))
            #StfConfig.log_op_list.append(print_op)
            ploss_pb = ploss_pb.dup_with_precision(x[0].fixedpoint)
            # print_obj += ("\n l230, backward_call_id=", backward_call_id, "\n ploss_pb= \n", tf.norm(ploss_pb.to_tf_tensor("R")))
            # print_op = tf.print(*# print_obj)
            # StfConfig.log_op_list.append(print_op)
            ploss_pw += [ploss_pb]
        backward_call_id +=1
        return ploss_pw, ploss_px

    def save(self, save_file_machine, sess, path):
        j = 0
        for weight in self.w:
            weight = weight.to_tf_tensor(owner=save_file_machine)
            weight = sess.run(weight)
            weight = pd.DataFrame(data=weight)
            weight.to_csv(path + "_{}".format(j), header=False, index=False)
            j += 1

    def load(self, path):
        j = 0
        w = []
        for weight in self.w:
            assert isinstance(weight, SharedVariablePair) or isinstance(weight, PrivateVariable)
            value = pd.read_csv(path + "_{}".format(j), header=None, index_col=None)
            value = np.array(value)
            value = np.reshape(value, weight.shape)
            weight.load_from_numpy(value, const=True)
            w += [weight]
            j += 1
        self.w = w







class Dense_Local(Layer):
    def __init__(self, output_dim, fathers, owner, with_b=True, activate=None):
        super(Dense_Local, self).__init__(output_dim=output_dim, fathers=fathers)
        self.w = []
        self.owner = get_device(owner)
        self.with_b = with_b
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")
            wi = PrivateVariable(owner=self.owner)
            # wi.from_numpy(np.random.uniform(size=[father.output_dim, output_dim],
            #                                 low=-1.0 / np.sqrt(father.output_dim + 1),
            #                                 high=1.0 / np.sqrt(father.output_dim + 1)))

            wi.load_from_numpy(
                np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))

            self.w += [wi]

        if with_b:
            b = PrivateVariable(owner=self.owner)
            b.load_from_numpy(np.zeros([output_dim]))
            self.w += [b]
        self.activate = activate
    def func(self, w: List[PrivateTensor], x: List[Union[PrivateTensor, SharedPair]]):

        if (len(w) != len(x)) and (len(w) != len(x) + 1):
            raise Exception("must have len(w) == len(x) or len(w)==len(x)+1")

        y = PrivateTensor.from_PrivteTensorBase(x[0].to_private(owner=self.owner), op_map=x[0].op_map) @ w[0]

        y = y.dup_with_precision(x[0].fixedpoint)

        for i in range(1, len(x)):
            xi = x[i].to_private(owner=self.owner)

            y = y + xi @ w[i]

        if self.with_b:
            y = y + self.w[len(x)]

        if self.activate is not None:
            if self.activate != "relu":
                raise Exception("only support for activate==relu")
            else:
                self.s = drelu_local(y)
                y = relu(y, self.s)

        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[PrivateTensor], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[PrivateTensor], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []

        ploss_py = ploss_py.to_private(owner=self.owner)
        ploss_py = PrivateTensor.from_PrivteTensorBase(ploss_py)

        if self.activate is not None:
            if self.activate != "relu":
                raise Exception("only support for activate==relu")
            else:
                ploss_py = relu_pull_back(None, ploss_py, drelu_b=self.s)

        for i in range(len(x)):
            ploss_pxi = ploss_py @ w[i].transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]

            xi = x[i].to_private(owner=self.owner)
            xi = PrivateTensor.from_PrivteTensorBase(xi)
            ploss_pwi = xi.transpose() @ ploss_py
            ploss_pwi = ploss_pwi.dup_with_precision(x[0].fixedpoint)
            ploss_pwi = ploss_pwi / batch_size
            ploss_pw += [ploss_pwi.dup_with_precision(x[0].fixedpoint)]

        ploss_px = dict(zip(self.fathers, list_ploss_px))
        if self.with_b:
            ploss_pb = ploss_py.reduce_sum(axis=[0]) / batch_size
            ploss_pw += [ploss_pb.dup_with_precision(x[0].fixedpoint)]

        return ploss_pw, ploss_px

    def save(self, save_file_machine, sess, path):
        j = 0
        for weight in self.w:
            weight = weight.to_tf_tensor(owner=save_file_machine)
            weight = sess.run(weight)
            weight = pd.DataFrame(data=weight)
            weight.to_csv(path + "_{}".format(j), header=False, index=False)
            j += 1

    def load(self, path):
        j = 0
        w = []
        for weight in self.w:
            assert isinstance(weight, SharedVariablePair) or isinstance(weight, PrivateVariable)
            value = pd.read_csv(path + "_{}".format(j), header=None, index_col=None)
            value = np.array(value)
            value = np.reshape(value, weight.shape)
            weight.load_from_numpy(value, const=True)
            w += [weight]
            j += 1
        self.w = w
