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
from stensorflow.basic.basic_class.private import PrivateTensor, PrivateVariable
from stensorflow.basic.basic_class.pair import SharedVariablePair, SharedPair
from stensorflow.basic.basic_class.base import get_device
from typing import Union, List
from stensorflow.basic.protocol.bilinear_triangle import BiliinearTriangle
import pandas as pd
from stensorflow.exception.exception import StfNoneException, StfCondException


class Dense_bak(Layer):
    """
    Dense Layer
    """
    def __init__(self, output_dim, fathers, with_b=True):
        if fathers is None:
            raise StfNoneException("fathers")
        if fathers == []:
            raise StfCondException("fathers != []", "fathers == []")
        super(Dense, self).__init__(output_dim=output_dim, fathers=fathers)
        self.with_b = with_b
        for father in fathers:
            if not isinstance(father, Layer):
                raise Exception("father must be a layer")
            wi = SharedVariablePair(ownerL="L", ownerR="R", shape=[father.output_dim, output_dim])
            wi.load_from_numpy(
                np.random.normal(scale=1.0 / np.sqrt(father.output_dim + 1), size=[father.output_dim, output_dim]))
            self.w += [wi]

        if with_b:
            b = SharedVariablePair(ownerL="L", ownerR="R", shape=[output_dim])
            b.load_from_numpy(np.zeros([output_dim]))
            self.w += [b]

    def __str__(self):
        return "Dense Layer of output_dim={}".format(self.output_dim)

    def __repr__(self):
        return self.__str__()

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if len(w) != len(x) + 1:
            raise Exception("must have len(w)==len(x)+1")

        y = x[0] @ w[0]
        y = y.dup_with_precision(x[0].fixedpoint)
        for i in range(1, len(x)):
            y = y + x[i] @ w[i]
        if self.with_b:
            y = y + self.w[len(x)]
        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []
        for i in range(len(x)):
            ploss_pxi = ploss_py @ w[i].transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]

            ploss_pwi = x[i].transpose() @ ploss_py
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


class Dense(Layer):
    """
    Dense Layer
    """
    def __init__(self, output_dim, fathers, with_b=True):
        if fathers is None:
            raise StfNoneException("fathers")
        if fathers == []:
            raise StfCondException("fathers != []", "fathers == []")
        super(Dense, self).__init__(output_dim=output_dim, fathers=fathers)
        self.with_b = with_b
        self.bt_list = []
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
            bt = BiliinearTriangle(f_xy, f_yz, f_zx) # x, w, plosspy^T
            self.bt_list.append(bt)

        if with_b:
            b = SharedVariablePair(ownerL="L", ownerR="R", shape=[output_dim])
            b.load_from_numpy(np.zeros([output_dim]))
            self.w += [b]

    def __str__(self):
        return "Dense Layer of output_dim={}".format(self.output_dim)

    def __repr__(self):
        return self.__str__()

    def func(self, w: List[SharedVariablePair], x: List[Union[PrivateTensor, SharedPair]]):
        if len(w) != len(x) + 1:
            raise Exception("must have len(w)==len(x)+1")

        # y = x[0] @ w[0]
        y = self.bt_list[0].compute_u(x[0], w[0])
        y = y.dup_with_precision(x[0].fixedpoint)
        for i in range(1, len(x)):
            # y = y + x[i] @ w[i]
            y = y + self.bt_list[i].compute_u(x[i], w[i])
        if self.with_b:
            y = y + self.w[len(x)]
        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[SharedPair], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[SharedPair], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []
        for i in range(len(x)):
            # ploss_pxi = ploss_py @ w[i].transpose()
            # ploss_pwi = x[i].transpose() @ ploss_py
            ploss_pxi_t, ploss_pwi_t = self.bt_list[i].compute_vw(ploss_py)
            ploss_pxi = ploss_pxi_t.transpose()
            ploss_pwi = ploss_pwi_t.transpose()
            list_ploss_px += [ploss_pxi.dup_with_precision(x[0].fixedpoint)]
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







class Dense_Local(Layer):
    def __init__(self, output_dim, fathers, owner, with_b=True):
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

        return y.dup_with_precision(new_fixedpoint=x[0].fixedpoint)

    def pull_back(self, w: List[PrivateTensor], x: List[Union[PrivateTensor, SharedPair]], y: SharedPair,
                  ploss_py: SharedPair) -> (List[PrivateTensor], List[SharedPair]):
        batch_size = x[0].shape[0]
        list_ploss_px = []
        ploss_pw = []

        ploss_py = ploss_py.to_private(owner=self.owner)
        ploss_py = PrivateTensor.from_PrivteTensorBase(ploss_py)

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
