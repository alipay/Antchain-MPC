#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : select
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-06-10 15:30
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.base import SharedTensorBase, SharedPairBase, PrivateTensorBase
from typing import Union
from stensorflow.basic.protocol.module_transform import module_transform
from stensorflow.basic.protocol.pm1_act import pm1_act, SharedTensorInt65, SharedPairInt65, pm1_pair_act_pair, \
    pm1_pair_act
from stensorflow.global_var import StfConfig
from stensorflow.exception.exception import StfCondException, StfEqualException, StfTypeException, StfValueException
import tensorflow as tf


def native_select(s: SharedPair, x: SharedPair, y: SharedPair = None, prf_flag=None,
                  compress_flag=None) -> SharedPair:
    """
     select using multiplication
    :param compress_flag:
    :param s: SharedPair  module=2
    :param x: SharedPai
    :param y: SharedPair, default=0
    :return: x if s else y
    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if s.xL.module != 2:
        raise StfValueException("s.module", 2, s.xL.module)
    if s.ownerL != x.ownerL or s.ownerR != x.ownerR:
        x = x.mirror()
    if s.ownerL != x.ownerL or s.ownerR != x.ownerR:
        raise Exception("must have s.ownerL == x.ownerL and s.ownerR == x.ownerR")

    if y is not None:
        if x.xL.module != y.xL.module:
            raise Exception("must have x.module==y.module")
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            y = y.mirror()
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            raise Exception("must have x.ownerL == y.ownerL and x.ownerR == y.ownerR")
        if not (s.shape == x.shape and s.shape == y.shape):
            raise Exception("must have  s.shape==x.shape and s.shape==y.shape")
        s = module_transform(s, x.xL.module, prf_flag=prf_flag, compress_flag=compress_flag)
        s = SharedPair.from_SharedPairBase(s)
        return s * (x - y) + y
    else:
        s = module_transform(s, x.xL.module, prf_flag=prf_flag, compress_flag=compress_flag)
        s = SharedPair.from_SharedPairBase(s)
        return s * x


def _select_sharedpair_sharedpair(s: SharedPairBase, x: SharedPairBase, y: SharedPairBase = None,
                                  prf_flag=None, compress_flag=None) -> SharedPair:
    """

    :param s:
    :param x:
    :param y:
    :return:  if y==None:
                sx
            else:
                s(x-y)+y

    """
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    if x.xL.module is not None:
        raise StfCondException("x.xL.module is None", "x.xL.module={}".format(x.xL.module))
    if s.xL.module != 2:
        raise StfValueException("s.xL.module", 2, s.xL.module)
    if s.ownerL != x.ownerL or s.ownerR != x.ownerR:
        x = x.mirror()
    if s.ownerL != x.ownerL or s.ownerR != x.ownerR:
        raise StfEqualException("s.owner", "x.owner", (s.ownerL, s.ownerR), (x.ownerL, x.ownerR))

    if y is None:
        if x.xL.module is not None and x.xL.module % 2 == 0:
            with tf.device(x.ownerL):
                xL = SharedTensorBase(inner_value=x.xL, module=2 * x.xL.module)
            with tf.device(x.ownerR):
                xR = SharedTensorBase(inner_value=x.xR, module=2 * x.xR.module)

            x_lift = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=xR, fixedpoint=x.fixedpoint)
            two_sx_lift = x_lift - pm1_pair_act_pair(s, x_lift, StfConfig.RS[0], prf_flag, compress_flag)
            with tf.device(x.ownerL):
                sxL = two_sx_lift.xL.inner_value // 2
                sxL = SharedTensorBase(inner_value=sxL, module=xL.module)

            with tf.device(x.ownerR):
                sxR = two_sx_lift.xR.inner_value - two_sx_lift.xR.inner_value // 2
                sxR = SharedTensorBase(inner_value=sxR, module=xR.module)

            sx = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=sxL, xR=sxR, fixedpoint=x.fixedpoint)

        elif x.xL.module is not None:
            inv_2 = x.xL.module - (x.xL.module // 2)
            two_sx = x - pm1_pair_act_pair(s, x, StfConfig.RS[0])
            sx = inv_2 * two_sx
        else:
            lift_x = SharedPairInt65.from_SharedPair(x)
            s_act_lift_x = pm1_pair_act_pair(s, lift_x, StfConfig.RS[0], prf_flag, compress_flag)
            two_sx = lift_x - s_act_lift_x
            sx = two_sx.half_of_even()
            sx = SharedPair.from_SharedPairBase(sx)
        return sx

    else:
        if x.xL.module != y.xL.module:
            raise StfEqualException("x.xL.module", "y.xL.module", x.xL.module, y.xL.module)
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            y = y.mirror()
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            raise StfEqualException("x.owner", "y.owner", (x.ownerL, x.ownerR), (y.ownerL, y.ownerR))
        if not (s.shape == x.shape and s.shape == y.shape):
            raise StfCondException("s.shape == x.shape and s.shape == y.shape",
                                   "s.shape={}, x.shape={}, y.shape={}".format(s.shape, x.shape, y.shape))
        z = x - y
        return _select_sharedpair_sharedpair(s, z) + y


def _select_private_private(s: PrivateTensor, x: PrivateTensor, y: PrivateTensor = None,
                            prf_flag=None, compress_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if s.module != 2:
        raise StfValueException("s.module", 2, s.module)
    if x.module is not None:
        raise StfCondException("x.module is None", "x.module={}".format(x.module))
    if y is None:
        if s.owner == x.owner:
            with tf.device(s.owner):
                s_mt = s.identity()
                s_mt.module = x.module
            return x * s_mt
        else:
            with tf.device(x.owner):
                lift_x = SharedTensorInt65.load_from_PrivateTensor(x)
            s_act_list_x = pm1_act(s.to_SharedTensor(), lift_x, s.owner, x.owner, RS_owner=StfConfig.RS[0],
                                   prf_flag=prf_flag, compress_flag=compress_flag)
            two_sx = SharedPairInt65.load_from_SharedTensorInt65(lift_x, owner=x.owner,
                                                                 other_owner=s.owner) - s_act_list_x
            sx = two_sx.half_of_even()
            sx.fixedpoint = x.fixedpoint
            return sx
    else:
        if x.owner == y.owner:
            z = _select_private_private(s=s, x=x - y)
        else:
            z = _select_private_sharedpair(s=s, x=x - y)
        if isinstance(z, SharedPairBase) and not isinstance(z, SharedPair):
            z = SharedPair.from_SharedPairBase(z)
        elif isinstance(z, PrivateTensorBase) and not isinstance(z, PrivateTensor):
            z = PrivateTensor.from_PrivteTensorBase(z)
        return z + y


def _select_private_sharedpair(s: PrivateTensor, x: SharedPair, y: SharedPair = None,
                               prf_flag=None, compress_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag

    if s.module != 2:
        raise StfValueException("s.module", 2, s.module)
    if s.shape != x.shape:
        raise StfEqualException("s.shape", "x.shape", s.shape, x.shape)
    if x.xL.module is not None:
        raise StfCondException("x.xL.module is None", "x.xL.module={}".format(x.xL.module))
    if y is None:
        xL = PrivateTensor(owner=x.ownerL, inner_value=x.xL.inner_value, fixedpoint=x.fixedpoint, module=x.xL.module)
        xR = PrivateTensor(owner=x.ownerR, inner_value=x.xR.inner_value, fixedpoint=x.fixedpoint, module=x.xR.module)
        zL = _select_private_private(s=s, x=xL, prf_flag=prf_flag, compress_flag=compress_flag)
        zR = _select_private_private(s=s, x=xR, prf_flag=prf_flag, compress_flag=compress_flag)
        return zL + zR
    else:
        if x.xL.module != y.xL.module:
            raise StfEqualException("x.xL.module", "y.xL.module", x.xL.module, y.xL.module)
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            y = y.mirror()
        if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
            raise StfEqualException("x.owner", "y.owner", (x.ownerL, x.ownerR), (y.ownerL, y.ownerR))
        if not (s.shape == x.shape and s.shape == y.shape):
            raise StfCondException("s.shape == x.shape and s.shape == y.shape",
                                   "s.shape={}, x.shape={}, y.shape={}".format(s.shape, x.shape, y.shape))
        z = x - y
        return _select_private_sharedpair(s, z) + y


def _select_sharedpair_private(s: SharedPairBase, x: PrivateTensorBase, y: PrivateTensorBase = None,
                               prf_flag=None, compress_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    if x.module is not None:
        raise StfCondException("x.module is None", "x.module={}".format(x.module))
    if s.xL.module != 2:
        raise StfValueException("s.xL.module", 2, s.xL.module)
    if s.shape != x.shape:
        raise StfEqualException("s.shape", "x.shape", s.shape, x.shape)
    if y is None:
        with tf.device(x.owner):
            lift_x = SharedTensorInt65.load_from_PrivateTensor(x)
        s_act_lift_x = pm1_pair_act(s, lift_x, x_owner=x.owner, RS_owner=StfConfig.RS[0],
                                    prf_flag=prf_flag, compress_flag=compress_flag)
        if s.ownerL == x.owner:
            other_owner = s.ownerR
        elif s.ownerR == x.owner:
            other_owner = s.ownerL
        else:
            raise StfCondException("x.owner==s.ownerL or x.owner==s.ownerR",
                                   "x.owner={}, s.ownerL={}, s.ownerR={}".format(x.owner, s.ownerL, s.ownerR))

        two_sx = SharedPairInt65.load_from_SharedTensorInt65(lift_x, owner=x.owner,
                                                             other_owner=other_owner) - s_act_lift_x
        sx = two_sx.half_of_even()
        sx.fixedpoint = x.fixedpoint
        return sx
    else:
        if x.module != y.module:
            raise StfEqualException("x.module", "y.module", x.module, y.module)
        if x.shape != y.shape:
            raise StfEqualException("x.shape", "y.shape", x.shape, y.shape)
        if x.owner == y.owner:
            z = x - y
            return _select_sharedpair_private(s, z) + y
        else:
            z = x - y
            return _select_sharedpair_sharedpair(s, z) + y


def select_share(s: Union[PrivateTensorBase, SharedPairBase, tf.Tensor], x: Union[PrivateTensorBase, SharedPairBase],
                 y: Union[PrivateTensorBase, SharedPairBase] = 0, prf_flag=None, compress_flag=None):
    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if compress_flag is None:
        compress_flag = StfConfig.compress_flag
    if y == 0:
        y = None

    if isinstance(s, SharedPairBase) and isinstance(x, SharedPairBase):
        return _select_sharedpair_sharedpair(s, x, y, prf_flag, compress_flag)
    elif isinstance(s, SharedPairBase) and isinstance(x, PrivateTensorBase):
        return _select_sharedpair_private(s, x, y, prf_flag, compress_flag)
    elif isinstance(s, PrivateTensorBase) and isinstance(x, PrivateTensorBase):
        return _select_private_private(s, x, y, prf_flag, compress_flag)
    elif isinstance(s, PrivateTensorBase) and isinstance(x, SharedPairBase):
        return _select_private_sharedpair(s, x, y, prf_flag, compress_flag)
    elif isinstance(s, tf.Tensor):
        if y is None:
            return s * x
        else:
            return s * (x - y) + y
    else:
        raise StfTypeException("s,x", "PrivateTensorBase or SharedPairBase", [type(s), type(x)])
