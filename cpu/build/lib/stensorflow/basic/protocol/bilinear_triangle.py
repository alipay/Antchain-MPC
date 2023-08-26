#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : bilinear_triangle
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2021/11/30 下午2:40
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from stensorflow.exception.exception import StfTypeException
from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.basic.basic_class.pair import SharedPair
import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.random.random import get_seed
from typing import Union

"""
(x~,y~)->u
(y~,z~)->v
(z~,x~)->w
"""


def BT_forward_SharedPair_SharedPair(x: SharedPairBase, y: SharedPairBase, f_xy, f_yz, f_zx, prf_flag=None):
    # f: (SharedTensorBase, SharedTensorBase)->SharedTensorBase

    if prf_flag is None:
        prf_flag = StfConfig.prf_flag

    if x.xL.module != y.xL.module:
        raise Exception("must have x.xL.module==y.xL.module")
    if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
        y = y.mirror()
    if x.ownerL != y.ownerL or x.ownerR != y.ownerR:
        raise Exception("must have x.ownerL==y.ownerL and x.ownerR==y.ownerR")
    with tf.device(StfConfig.RS[0]):
        if prf_flag:
            seed_xL = get_seed()
            seed_xR = get_seed()
            seed_yL = get_seed()
            seed_yR = get_seed()
            seed_zL = get_seed()
            seed_zR = get_seed()
            seed_u = get_seed()
            seed_v = get_seed()
            seed_w = get_seed()
        else:
            seed_xL = None
            seed_xR = None
            seed_yL = None
            seed_yR = None
            seed_zL = None
            seed_zR = None
            seed_u = None
            seed_v = None
            seed_w = None

        xL_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xL)
        xR_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_xR)
        yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yL)
        yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yR)
        u = f_xy(xL_adjoint + xR_adjoint, yL_adjoint + yR_adjoint)  # SharedTensor
        zL_adjoint = u.random_uniform_adjoint(seed_zL)
        zR_adjoint = u.random_uniform_adjoint(seed_zR)
        v = f_yz(yL_adjoint + yR_adjoint, zL_adjoint + zR_adjoint)  # SharedTensor
        w = f_zx(zL_adjoint + zR_adjoint, xL_adjoint + xR_adjoint)

        uL = u.random_uniform_adjoint(seed_u)
        uR = u - uL
        vL = v.random_uniform_adjoint(seed_v)
        vR = v - vL
        wL = w.random_uniform_adjoint(seed_w)
        wR = w - wL

    with tf.device(x.ownerL):
        delta_xL = x.xL - xL_adjoint
        delta_yL = y.xL - yL_adjoint
    with tf.device(y.ownerR):
        delta_xR = x.xR - xR_adjoint
        delta_yR = y.xR - yR_adjoint
    with tf.device(x.ownerL):
        delta_x_onL = delta_xL + delta_xR
        delta_y_onL = delta_yL + delta_yR
        rL = f_xy(delta_x_onL, y.xL) + f_xy(xL_adjoint, delta_y_onL) + uL
    with tf.device(x.ownerR):
        delta_x_onR = delta_xL + delta_xR
        delta_y_onR = delta_yL + delta_yR
        rR = f_xy(delta_x_onR, y.xR) + f_xy(xR_adjoint, delta_y_onR) + uR
    r = SharedPairBase(xL=rL, xR=rR, ownerL=x.ownerL, ownerR=x.ownerR, fixedpoint=x.fixedpoint + y.fixedpoint)
    if prf_flag:
        s = (delta_x_onL, delta_x_onR, delta_y_onL, delta_y_onR,
             seed_xL, seed_xR, seed_yL, seed_yR, seed_zL, seed_zR,
             seed_v, vR, seed_w, wR, x.fixedpoint, y.fixedpoint)
    else:
        s = (delta_x_onL, delta_x_onR, delta_y_onL, delta_y_onR,
            xL_adjoint, xR_adjoint, yL_adjoint, yR_adjoint, zL_adjoint, zR_adjoint,
            vL, vR, wL, wR, x.fixedpoint, y.fixedpoint)
    return r, s


def BT_backward_SharedPair_SharedPair(s, z: SharedPairBase, f_yz, f_zx, prf_flag):
    if prf_flag:
        delta_x_onL, delta_x_onR, delta_y_onL, delta_y_onR, \
        seed_xL, seed_xR, seed_yL, seed_yR, seed_zL, seed_zR, \
        seed_v, vR, seed_w, wR, x_fixedpoint, y_fixedpoint = s
        with tf.device(z.ownerL):
            xL_adjoint = delta_x_onL.random_uniform_adjoint(seed_xL)
            yL_adjoint = delta_y_onL.random_uniform_adjoint(seed_yL)
            zL_adjoint = z.to_SharedTensor_like().random_uniform_adjoint(seed_zL)
            vL = vR.random_uniform_adjoint(seed_v)
            wL = wR.random_uniform_adjoint(seed_w)
        with tf.device(z.ownerR):
            xR_adjoint = delta_x_onR.random_uniform_adjoint(seed_xR)
            yR_adjoint = delta_y_onR.random_uniform_adjoint(seed_yR)
            zR_adjoint = z.to_SharedTensor_like().random_uniform_adjoint(seed_zR)
    else:
        delta_x_onL, delta_x_onR, delta_y_onL, delta_y_onR, \
        xL_adjoint, xR_adjoint, yL_adjoint, yR_adjoint, zL_adjoint, zR_adjoint, \
        vL, vR, wL, wR, x_fixedpoint, y_fixedpoint = s
    with tf.device(z.ownerL):
        delta_zL = z.xL - zL_adjoint
    with tf.device(z.ownerR):
        delta_zR = z.xR - zR_adjoint
    with tf.device(z.ownerL):
        delta_z_onL = delta_zL+delta_zR
        vL = f_yz(delta_y_onL, z.xL) + f_yz(yL_adjoint, delta_z_onL) + vL
        wL = f_zx(z.xL, delta_x_onL) + f_zx(delta_z_onL, xL_adjoint) + wL
    with tf.device(z.ownerR):
        delta_z_onR = delta_zL+delta_zR
        vR = f_yz(delta_y_onR, z.xR) + f_yz(yR_adjoint, delta_z_onR) + vR
        wR = f_zx(z.xR, delta_x_onR) + f_zx(delta_z_onR, xR_adjoint) + wR
    v = SharedPairBase(xL=vL, xR=vR, ownerL=z.ownerL, ownerR=z.ownerR, fixedpoint=y_fixedpoint + z.fixedpoint)
    w = SharedPairBase(xL=wL, xR=wR, ownerL=z.ownerL, ownerR=z.ownerR, fixedpoint=z.fixedpoint + x_fixedpoint)
    return v, w



def BT_forward_PrivateTensor_SharedPair(x: PrivateTensorBase, y: SharedPairBase, f_xy, f_yz, f_zx, prf_flag=None):
    # f: (PrivateTensorBase, SharedTensorBase)->SharedTensorBase

    if prf_flag is None:
        prf_flag = StfConfig.prf_flag
    if x.module != y.xL.module:
        raise Exception("must have x.module==y.module")
    if y.ownerL == y.ownerR:
        raise Exception("must have y.ownerL!=y.ownerR")
    if x.owner == y.ownerL:
        with tf.device(StfConfig.RS[0]):
            if prf_flag:
                seed_x = get_seed()
                seed_yL = get_seed()
                seed_yR = get_seed()
                seed_zL = get_seed()
                seed_zR = get_seed()
                seed_u = get_seed()
                seed_v = get_seed()
                seed_w = get_seed()
            else:
                seed_x = None
                seed_yL = None
                seed_yR = None
                seed_zL = None
                seed_zR = None
                seed_u = None
                seed_v = None
                seed_w = None
            x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_x)
            yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yL)
            yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yR)
            u = f_xy(x_adjoint, yL_adjoint + yR_adjoint)  # SharedTensor
            zL_adjoint = u.random_uniform_adjoint(seed_zL)
            zR_adjoint = u.random_uniform_adjoint(seed_zR)
            v = f_yz(yL_adjoint + yR_adjoint, zL_adjoint + zR_adjoint)  # SharedTensor
            w = f_zx(zL_adjoint + zR_adjoint, x_adjoint)

            uL = u.random_uniform_adjoint(seed_u)
            uR = u - uL
            vL = v.random_uniform_adjoint(seed_v)
            vR = v - vL
            wL = w.random_uniform_adjoint(seed_w)
            wR = w - wL

        with tf.device(x.owner):
            if prf_flag:
                x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_x)
                yL_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yL)
            delta_x = x.to_SharedTensor() - x_adjoint
            delta_yL = y.xL - yL_adjoint
        with tf.device(y.ownerR):
            if prf_flag:
                yR_adjoint = y.to_SharedTensor_like().random_uniform_adjoint(seed_yR)
            delta_yR = y.xR - yR_adjoint
        with tf.device(x.owner):
            delta_y_onL = delta_yL + delta_yR
            rL = f_xy(delta_x, y.xL) + f_xy(x_adjoint, delta_y_onL) + uL
        with tf.device(y.ownerR):
            delta_y_onR = delta_yL + delta_yR
            rR = f_xy(delta_x, y.xR) + uR
        r = SharedPairBase(xL=rL, xR=rR, ownerL=x.owner, ownerR=y.ownerR, fixedpoint=x.fixedpoint + y.fixedpoint)
        if prf_flag:
            s = (delta_x, delta_y_onL, delta_y_onR,
                 seed_x, seed_yL, seed_yR, seed_zL, seed_zR,
                 seed_v, vR, seed_w, wR, x.fixedpoint, y.fixedpoint)
        else:
            s = (delta_x, delta_y_onL, delta_y_onR,
                x_adjoint, yL_adjoint, yR_adjoint, zL_adjoint, zR_adjoint,
                vL, vR, wL, wR, x.fixedpoint, y.fixedpoint)
    elif x.owner == y.ownerR:
        y = y.mirror()
        r, s = BT_forward_PrivateTensor_SharedPair(x, y, f_xy, f_yz, f_zx, prf_flag=prf_flag)
    else:
        raise Exception("must have x.owner==y.ownerL or x.owner==y.ownerR")
    return r, s




def BT_backward_PrivateTensor_SharedPair(s, z: SharedPairBase, f_yz, f_zx, x: PrivateTensorBase, prf_flag=None):

    if prf_flag:
        delta_x, delta_y_onL, delta_y_onR, \
         seed_x, seed_yL, seed_yR, seed_zL, seed_zR, \
         seed_v, vR, seed_w, wR, x_fixedpoint, y_fixedpoint = s
        with tf.device(z.ownerL):
            x_adjoint = x.to_SharedTensor_like().random_uniform_adjoint(seed_x)
            yL_adjoint = delta_y_onL.random_uniform_adjoint(seed_yL)
            zL_adjoint = z.to_SharedTensor_like().random_uniform_adjoint(seed_zL)
            vL = vR.random_uniform_adjoint(seed_v)
            wL = wR.random_uniform_adjoint(seed_w)
        with tf.device(z.ownerR):
            yR_adjoint = delta_y_onR.random_uniform_adjoint(seed_yR)
            zR_adjoint = z.to_SharedTensor_like().random_uniform_adjoint(seed_zR)
    else:
        delta_x, delta_y_onL, delta_y_onR, \
        x_adjoint, yL_adjoint, yR_adjoint, zL_adjoint, zR_adjoint, \
        vL, vR, wL, wR, x_fixedpoint, y_fixedpoint = s
    with tf.device(z.ownerL):
        delta_zL = z.xL - zL_adjoint
    with tf.device(z.ownerR):
        delta_zR = z.xR - zR_adjoint
    with tf.device(z.ownerL):
        delta_z_onL = delta_zL+delta_zR
        vL = f_yz(delta_y_onL, z.xL) + f_yz(yL_adjoint, delta_z_onL) + vL
        if x.owner == z.ownerL:
            wL = f_zx(z.xL, delta_x) + f_zx(delta_z_onL, x_adjoint) + wL
        elif x.owner == z.ownerR:
            wL = f_zx(z.xL, delta_x) + wL
        else:
            raise Exception("must have x.owner == z.ownerL or x.owner == z.ownerR")
    with tf.device(z.ownerR):
        delta_z_onR = delta_zL+delta_zR
        vR = f_yz(delta_y_onR, z.xR) + f_yz(yR_adjoint, delta_z_onR) + vR
        if x.owner == z.ownerL:
            wR = f_zx(z.xR, delta_x) + wR
        elif x.owner == z.ownerR:
            wR = f_zx(z.xR, delta_x) + f_zx(delta_z_onR, x_adjoint) + wR
        else:
            raise Exception("must have x.owner == z.ownerL or x.owner == z.ownerR")
    v = SharedPairBase(xL=vL, xR=vR, ownerL=z.ownerL, ownerR=z.ownerR, fixedpoint=y_fixedpoint + z.fixedpoint)
    w = SharedPairBase(xL=wL, xR=wR, ownerL=z.ownerL, ownerR=z.ownerR, fixedpoint=z.fixedpoint + x_fixedpoint)
    return v, w



def BT_forward_SharedPair_PrivateTensor(x: SharedPairBase, y: PrivateTensorBase, f_xy, f_yz, f_zx, prf_flag=None):
    x1 = y
    y1 = x
    g_X1Y1 = lambda a, b: f_xy(b, a)
    g_Y1Z = lambda a, b: f_zx(b, a)
    g_ZX1 = lambda a, b: f_yz(b, a)
    return BT_forward_PrivateTensor_SharedPair(x1, y1, g_X1Y1, g_Y1Z, g_ZX1, prf_flag)


def BT_backward_SharedPair_PrivateTensor(s, z: SharedPairBase, f_yz, f_zx, y: PrivateTensorBase, prf_flag=None):
    x1 = y
    g_Y1Z = lambda a, b: f_zx(b, a)
    g_ZX1 = lambda a, b: f_yz(b, a)
    u, v = BT_backward_PrivateTensor_SharedPair(s, z, g_Y1Z, g_ZX1, x1, prf_flag)
    return v, u

class BiliinearTriangle():
    """
    (x,y)->u
    (y,z)->v
    (z,x)->w
    """
    def __init__(self, f_xy, f_yz, f_zx):
        self.f_xy = f_xy
        self.f_yz = f_yz
        self.f_zx = f_zx
        self.s = None
        self.x = None
        self.y = None
        self.prf_flag = StfConfig.prf_flag

    def compute_u(self, x: Union[PrivateTensorBase, SharedPairBase], y: Union[PrivateTensorBase, SharedPairBase]):
        self.x = x
        self.y = y
        is_private_x = isinstance(x, PrivateTensorBase)
        is_private_y = isinstance(y, PrivateTensorBase)
        if is_private_x and is_private_y:
            raise NotImplementedError
        elif is_private_x and not is_private_y:
            r, s = BT_forward_PrivateTensor_SharedPair(x, y, self.f_xy, self.f_yz, self.f_zx, self.prf_flag)
        elif not is_private_x and is_private_y:
            r, s = BT_forward_SharedPair_PrivateTensor(x, y, self.f_xy, self.f_yz, self.f_zx, self.prf_flag)
        else:
            r, s = BT_forward_SharedPair_SharedPair(x, y, self.f_xy, self.f_yz, self.f_zx, self.prf_flag)
        self.s = s
        return SharedPair.from_SharedPairBase(r)

    def compute_vw(self, z: SharedPairBase):
        is_private_x = isinstance(self.x, PrivateTensorBase)
        is_private_y = isinstance(self.y, PrivateTensorBase)

        if is_private_x and is_private_y:
            raise NotImplementedError
        elif is_private_x and not is_private_y:
            u, v = BT_backward_PrivateTensor_SharedPair(self.s, z, self.f_yz, self.f_zx, self.x,self.prf_flag)
        elif not is_private_x and is_private_y:
            u, v = BT_backward_SharedPair_PrivateTensor(self.s, z, self.f_yz, self.f_zx, self.y, self.prf_flag)
        else:
            u, v = BT_backward_SharedPair_SharedPair(self.s, z, self.f_yz, self.f_zx, self.prf_flag)
        return SharedPair.from_SharedPairBase(u), SharedPair.from_SharedPairBase(v)

