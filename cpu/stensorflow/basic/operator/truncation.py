#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : trancation
   Author : Qizhi Zhang
   Email: zqz.math@gmailcom
   Create Time : 2022-12-25  22:32
   Description :  exact truncation
"""
import warnings

from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedPairBase, SharedTensorBase
from typing import Union
from stensorflow.basic.protocol.bilinear_map import BM_PrivateTensor_PrivateTensor
import tensorflow as tf
import random
from stensorflow.basic.protocol.module_transform import module_transform
from stensorflow.global_var import StfConfig



# def dup_with_precision(x: SharedPair, new_fixepoint, non_negative=False):
#     if new_fixepoint>=x.fixedpoint:
#         return x.dup_with_precision(new_fixepoint)
#     elif non_negative:
#         carry = get_carry(x, non_negative)
#         d = (x.fixedpoint - new_fixepoint)
#         with tf.device(x.ownerL):
#             xL = (x.xL >> d) % (1<<(64-d))
#             xL.module = None
#         with tf.device(x.ownerR):
#             x0R = (x.xR >> d) % (1<<(64-d))
#             x0R.module = None
#             x1R = ((-x.xR) >> d) % (1<<(64-d))
#             x1R.module = None
#             x1R = -x1R
#
#         y0 = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=x0R, fixedpoint=new_fixepoint, op_map=x.op_map)
#         y1 = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=x1R, fixedpoint=new_fixepoint, op_map=x.op_map)
#         y = select_share(carry, y1, y0)
#         return y
#     else:
#         raise NotImplementedError







        # d = (x.fixedpoint - new_fixepoint)
        # with tf.device(x.ownerL):
        #     xL_top = (x.xL >> 62) % 4
        #     xLa = x.xL>>d
        #     xLb = ((x.xL + (1<<62)) + (1<<62))>>d
        #
        # with tf.device(x.ownerR):
        #     mxR = -x.xR
        #     mxR_top = (mxR >>62) % 4
        #     mxRa = mxR>>d
        #     mxRb = ((mxR + (1<<62)) + (1<<62))>>d
        #
        # xa = SharedPair(x.ownerL, x.ownerR, xL=xLa, xR=-mxRa, fixedpoint=new_fixepoint)
        # xb = SharedPair(x.ownerL, x.ownerR, xL=xLb, xR=-mxRb, fixedpoint=new_fixepoint)
        #
        # def is_10_01_or_01_10(xL_top, xR_top):
        #     """
        #     [xL_top, x_R_top]==[[1,0], [0,1]] or [[0,1], [1,0]] then return true  else false
        #     """
        #     with tf.device(x.ownerL):
        #         xL_top |= 2
        #     with tf.device(x.ownerR):
        #         xR_top |= 1
        #
        #     # [xL_top, x_R_top]==[[0,0],[0,0]] or [[1,1], [1,1]] then return true  else false
        #
        #     with tf.device(x.ownerL):
        #         xL_top_0 = xL_top & 1
        #         xL_top_1 = (xL_top & 2) >> 1
        #         yL = xL_top_0 + xL_top_1
        #         yL.module = 4
        #     with tf.device(x.ownerR):
        #         xR_top_0 = xR_top & 1
        #         xR_top_1 = (xR_top & 2) >> 1
        #         yR = xR_top_0 + xR_top_1
        #         yR.module = 4
        #
        #     # xL_top_0+ xL_top_1 + xR_top_0 + xR_top_1 =0 mod 4 then return true  else false
        #     y = SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=yL, xR=yR, fixedpoint=0)
        #     z = isZero(y)
        #     return SharedPair.from_SharedPairBase(z, op_map=x.op_map)
        #
        # s = is_10_01_or_01_10(xL_top, xR_top=mxR_top)
        #
        # return select_share(s, xb, xa)



truncation_id =0

def dup_with_precision(x: Union[SharedPairBase, PrivateTensorBase], new_fixepoint, non_negative=False):
    global truncation_id
    if isinstance(x, PrivateTensorBase):
        return x.dup_with_precision(new_fixepoint)
    elif isinstance(x, SharedPairBase) and new_fixepoint>=x.fixedpoint:
        return SharedPairBase.dup_with_precision(x, new_fixepoint)
    elif StfConfig.truncation_functionality:
        warnings.warn("Now a functionality for truncation is used")
        xL = x.to_private("L")
        # xR = x.to_private("R")
        yL = xL.dup_with_precision(new_fixepoint).to_SharedTensor()
        # yR = xR.dup_with_precision(new_fixepoint)
        # yR = x.xR.zeros_like()

        yR = x.xR.random_uniform_adjoint()%2
        yR.module = None

        # z = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, shape=x.shape, fixedpoint=new_fixepoint,
        #                    xL=yL.to_SharedTensor(), xR=yR.to_SharedTensor().zeros_like())
        z = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, shape=x.shape, fixedpoint=new_fixepoint,
                           xL=yL, xR=yR)

        #z = xL.to_SharedPairBase(other_owner="R")
        return z


        # print_op = tf.print("l116, truncation_id=", truncation_id, "\n x=", tf.norm(x.to_tf_tensor("R")), "\n z=", tf.norm(z.to_tf_tensor("R")))
        # StfConfig.log_op_list.append(print_op)
        # truncation_id +=1
        # y = x.to_tf_tensor("R")
        # z = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, shape=x.shape, fixedpoint=new_fixepoint)
        # z.load_from_tf_tensor(y)
        # d = (x.fixedpoint - new_fixepoint)
        # with tf.device(StfConfig.RS[0]):
        #     rPrimeL = x.xL.random_uniform_adjoint()
        #     rPrime = rPrimeL.ones_like() << d
        #     r = rPrime.ones_like()
        #     rPrimeR = rPrime - rPrimeL
        #     rL = x.xL.random_uniform_adjoint()
        #     rR = r - rL
        # with tf.device(x.ownerL):
        #     deltaxL = x.xL - rPrimeL
        # with tf.device(x.ownerR):
        #     deltaxR = x.xR - rPrimeR
        # with tf.device(x.ownerL):
        #     zL = ((deltaxL+deltaxR) >> d) + rL
        # with tf.device(x.ownerR):
        #     zR = ((deltaxL+deltaxR) >> d) + rR
        # z = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=zL, xR=zR, fixedpoint=new_fixepoint)
        # return z
    else:
        r = StfConfig.truncation_without_error_ratio
        if random.random() <= r:
            pass
        else:
            return SharedPairBase.dup_with_precision(x, new_fixepoint)

    if non_negative:
        dup_with_precision.call_times_non_negative +=1
        d = (x.fixedpoint - new_fixepoint)
        with tf.device(x.ownerL):
            xL_top = (x.xL >> 62) % 4
            xL_top0 = (xL_top >> 1) % 2
            L_in1 = xL_top0
            xL = x.xL>>d
            L_in1 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in1.inner_value, module=2)

        with tf.device(x.ownerR):
            mxR = -x.xR
            mxR_top = (mxR >> 62) % 4
            mxR_top0 = (mxR_top >> 1) % 2
            R_in0 = mxR_top0 + mxR_top0.ones_like()
            mxR = mxR>>d
            R_in0 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in0.inner_value, module=2)

        # L_in1_R_in0 = mul(L_in1, R_in0)
        L_in1_R_in0 = BM_PrivateTensor_PrivateTensor(L_in1, R_in0, lambda _x, _y: SharedTensorBase.__mul__(_x, _y))
        L_in1_R_in0 = module_transform(L_in1_R_in0, new_module=(1<<d))
        L_in1_R_in0.fixedpoint = new_fixepoint
        with tf.device(x.ownerL):
            L_in1_R_in0.xL.module = None
        with tf.device(x.ownerR):
            L_in1_R_in0.xR.module = None

        y = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=-mxR, fixedpoint=new_fixepoint)
        y = y + (L_in1_R_in0 << (64-d))
        return y
    else:
        dup_with_precision.call_times_general +=1
        # print("exact truncation for all number")
        d = (x.fixedpoint - new_fixepoint)
        with tf.device(x.ownerL):
            xL_top = (x.xL >> 62) % 4
            xL_top1 = xL_top % 2
            xL_top0 = (xL_top >> 1) % 2
            L_in10 = xL_top0 * (xL_top1 + xL_top1.ones_like())
            L_in01 = (xL_top0 + xL_top1.ones_like()) * xL_top1
            xL = x.xL >> d
            L_in10 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in10.inner_value, module=2)
            L_in01 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in01.inner_value, module=2)

        with tf.device(x.ownerR):
            mxR = -x.xR
            mxR_top = (mxR >> 62) % 4
            mxR_top1 = mxR_top % 2
            mxR_top0 = (mxR_top >> 1) % 2
            R_in01 = (mxR_top0 + mxR_top0.ones_like()) * mxR_top1
            R_in10 = mxR_top0 * (mxR_top1 + mxR_top0.ones_like())
            mxR = mxR >> d
            R_in01 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in01.inner_value, module=2)
            R_in10 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in10.inner_value, module=2)


        # L_in10_R_in01 = mul(L_in10, R_in01)
        L_in10_R_in01 = BM_PrivateTensor_PrivateTensor(L_in10, R_in01, lambda _x, _y: SharedTensorBase.__mul__(_x, _y))
        L_in10_R_in01 = module_transform(L_in10_R_in01, new_module=(1 << d))
        L_in10_R_in01.fixedpoint = new_fixepoint
        with tf.device(x.ownerL):
            L_in10_R_in01.xL.module = None
        with tf.device(x.ownerR):
            L_in10_R_in01.xR.module = None

        # L_in01_R_in10 = mul(L_in01, R_in10)
        L_in01_R_in10 = BM_PrivateTensor_PrivateTensor(L_in01, R_in10, lambda _x, _y: SharedTensorBase.__mul__(_x, _y))
        L_in01_R_in10 = module_transform(L_in01_R_in10, new_module=(1<<d))
        L_in01_R_in10.fixedpoint = new_fixepoint
        with tf.device(x.ownerL):
            L_in01_R_in10.xL.module = None
        with tf.device(x.ownerR):
            L_in01_R_in10.xR.module = None

        y = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=-mxR, fixedpoint=new_fixepoint)
        y = y + (L_in10_R_in01 << (64 - d)) - (L_in01_R_in10 << (64-d))
        # print("y.fixepoint=", y.fixedpoint)
        return y


dup_with_precision.call_times_non_negative = 0
dup_with_precision.call_times_general = 0



# error
# def dup_with_precision(x: SharedPairBase, new_fixepoint, non_negative=False):
#     if new_fixepoint>=x.fixedpoint:
#         raise Exception("only for new_fixepoint<x.fixedpoint")
#     elif non_negative:
#         d = (x.fixedpoint - new_fixepoint)
#         with tf.device(x.ownerL):
#             xL_top = x.xL >> 62
#             xL_top0 = (xL_top >> 1) % 2
#             #xL_top0 = (x.xL >> 63) % 2
#             L_in1 = xL_top0
#             print_op = [tf.print("xL=", x.xL.inner_value)]
#             StfConfig.log_op_list.append(print_op)
#             xL = x.xL>>1
#             L_in1 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in1.inner_value, module=2)
#
#         with tf.device(x.ownerR):
#             mxR = -x.xR
#             mxR_top = mxR >> 62
#             mxR_top0 = (mxR_top >> 1) % 2
#             # mxR_top0 = (mxR >> 63) % 2
#             R_in0 = mxR_top0 + mxR_top0.ones_like()
#             print_op = [tf.print("mxR=", mxR.inner_value)]
#             StfConfig.log_op_list.append(print_op)
#             mxR = mxR>>1
#             R_in0 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in0.inner_value, module=2)
#
#         L_in1_R_in0 = mul(L_in1, R_in0)
#         # L_in1_R_in0 = module_transform(L_in1_R_in0, new_module=(1<<d))
#         L_in1_R_in0.fixedpoint = new_fixepoint
#         with tf.device(x.ownerL):
#             L_in1_R_in0.xL.module = None
#         with tf.device(x.ownerR):
#             L_in1_R_in0.xR.module = None
#         print_op = tf.print("l217")
#         StfConfig.log_op_list.append(print_op)
#         print_op = tf.print("\n L_in1_R_in0=", L_in1_R_in0.xL.inner_value+L_in1_R_in0.xR.inner_value)
#         StfConfig.log_op_list.append(print_op)
#         with tf.device(x.ownerL):
#             xL = (xL + (L_in1_R_in0.xL << 63)) >> (d-1)
#         with tf.device(x.ownerR):
#             mxR = (mxR + (L_in1_R_in0.xR << 63)) >> (d-1)
#         print_op = tf.print("l240: xL=", xL.inner_value, "\n mxR=", mxR.inner_value)
#         StfConfig.log_op_list.append(print_op)
#         y = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=-mxR, fixedpoint=new_fixepoint)
#         return y
#     else:
#         print("exact truncation for any number")
#         d = (x.fixedpoint - new_fixepoint)
#         with tf.device(x.ownerL):
#             xL_top = (x.xL >> 62) % 4
#             xL_top1 = xL_top % 2
#             xL_top0 = (xL_top >> 1) % 2
#             L_in10 = xL_top0 * (xL_top1 + xL_top1.ones_like())
#             L_in01 = (xL_top0 + xL_top1.ones_like()) * xL_top1
#             xL = x.xL >> 1
#             L_in10 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in10.inner_value, module=2)
#             L_in01 = PrivateTensorBase(owner=x.ownerL, fixedpoint=0, inner_value=L_in01.inner_value, module=2)
#
#         with tf.device(x.ownerR):
#             mxR = -x.xR
#             mxR_top = (mxR >> 62) % 4
#             mxR_top1 = mxR_top % 2
#             mxR_top0 = (mxR_top >> 1) % 2
#             R_in01 = (mxR_top0 + mxR_top0.ones_like()) * mxR_top1
#             R_in10 = mxR_top0 * (mxR_top1 + mxR_top0.ones_like())
#             mxR = mxR >> 1
#             R_in01 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in01.inner_value, module=2)
#             R_in10 = PrivateTensorBase(owner=x.ownerR, fixedpoint=0, inner_value=R_in10.inner_value, module=2)
#
#
#         L_in10_R_in01 = mul(L_in10, R_in01)
#         # L_in10_R_in01 = module_transform(L_in10_R_in01, new_module=(1 << d))
#         L_in10_R_in01.fixedpoint = new_fixepoint
#         with tf.device(x.ownerL):
#             L_in10_R_in01.xL.module = None
#         with tf.device(x.ownerR):
#             L_in10_R_in01.xR.module = None
#
#         L_in01_R_in10 = mul(L_in01, R_in10)
#         # L_in01_R_in10 = module_transform(L_in01_R_in10, new_module=(1<<d))
#         L_in01_R_in10.fixedpoint = new_fixepoint
#         with tf.device(x.ownerL):
#             L_in01_R_in10.xL.module = None
#         with tf.device(x.ownerR):
#             L_in01_R_in10.xR.module = None
#
#         y = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=-mxR, fixedpoint=new_fixepoint)
#         y = y + (L_in10_R_in01 << 63) - (L_in01_R_in10 << 63)
#         # print("y.fixepoint=", y.fixedpoint)
#         if d>1:
#             with tf.device(y.ownerL):
#                 xL = y.xL >> (d-1)
#             with tf.device(y.ownerR):
#                 xR = -((-y.xR) >> (d-1))
#             y = SharedPairBase(ownerL=x.ownerL, ownerR=x.ownerR, xL=xL, xR=xR, fixedpoint=new_fixepoint)
#         return y
#
