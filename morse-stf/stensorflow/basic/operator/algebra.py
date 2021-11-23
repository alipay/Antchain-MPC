#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : algebra
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-08-03 20:00
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import PrivateTensorBase, SharedTensorBase
from stensorflow.basic.basic_class.base import SharedPairBase
# from stensorflow.basic.basic_class.private import PrivateTensor
from stensorflow.global_var import StfConfig
import tensorflow as tf
from stensorflow.exception.exception import StfCondException


def concat(x: list, axis, fixed_point=None):
    """

    :param x:
    :param axis:
    :return:
    """
    is_SharedTensorBase = list(map(lambda u: isinstance(u, SharedTensorBase), x))
    if all(is_SharedTensorBase):  # all x are SharedTensorBase
        if len(set(map(lambda u: u.module, x))) > 1:
            raise Exception("all x must have same module")
        module = x[0].module
        inner_value = tf.concat(list(map(lambda u: u.inner_value, x)), axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=module)
    elif any(is_SharedTensorBase):
        raise Exception("must all x are SharedTensorBase or none of x is SharedTensorBase")

    if fixed_point is None:
        fixed_point = max(min(list(map(lambda u: u.fixedpoint, x))), StfConfig.default_fixed_point)
    x = list(map(lambda u: u.dup_with_precision(fixed_point), x))

    if all(list(map(lambda u: isinstance(u, PrivateTensorBase), x))):  # all x are PrivateTensorBase
        if len(set(map(lambda u: u.owner, x))) == 1:  # all x have same owner
            if len(set(map(lambda u: u.module, x))) > 1:
                raise Exception("all x must have same module")
            # fixed_point = min(list(map(lambda u: u.fixedpoint, x)))
            # x = list(map(lambda u: u.dup_with_precision(fixed_point), x))
            x_tf_tensor = list(map(lambda u: u.inner_value, x))
            y = tf.concat(x_tf_tensor, axis=axis)
            return PrivateTensorBase(owner=x[0].owner, fixedpoint=fixed_point, inner_value=y, module=x[0].module)
        else:
            owners = list(set(map(lambda u: u.owner, x)))
            if len(owners) > 2:
                raise StfCondException("len(owners)<=2", "owners={}".format(owners))

            x = list(
                map(
                    lambda u: u.to_SharedPairBase(
                        other_owner=owners[0] if owners[1] == u.owner else owners[1]),
                    x)
            )
            return concat(x, axis=axis, fixed_point=fixed_point)
    elif all(list(map(lambda u: isinstance(u, SharedPairBase), x))):  # all x[i] are SharedPairBase
        # y = x[0]
        # for i in range(1, len(x)):
        #     y = y.concat(x[i], axis=axis)
        # return y
        ownerL = x[0].ownerL
        ownerR = x[0].ownerR
        x = list(map(lambda u: u if u.ownerL == ownerL and u.ownerR == ownerR else u.mirror(), x))

        yL = concat(list(map(lambda u: u.xL, x)), axis=axis)
        yR = concat(list(map(lambda u: u.xR, x)), axis=axis)
        return SharedPairBase(ownerL=ownerL, ownerR=ownerR, xL=yL, xR=yR, fixedpoint=fixed_point)

    elif any(list(map(lambda u: isinstance(u, SharedPairBase), x))):  # exists x[i] is SharedPairBase
        for u in x:
            if isinstance(u, SharedPairBase):
                ownerL = u.ownerL
                ownerR = u.ownerR
                break
        for i in range(len(x)):
            if isinstance(x[i], PrivateTensorBase):
                if x[i].owner == ownerL:
                    other_owner = ownerR
                elif x[i].owner == ownerR:
                    other_owner = ownerL
                else:
                    raise Exception("must have x[i].owner == ownerL or x[i].owner == ownerR")
                x[i] = x[i].to_SharedPairBase(other_owner=other_owner)
        return concat(x, axis)
    else:

        raise Exception("x[i] must be PrivateTensorBase or SharedPairBase")

def stack(x: list, axis, fixed_point=None):
    """

    :param x:
    :param axis:
    :return:
    """
    is_SharedTensorBase = list(map(lambda u: isinstance(u, SharedTensorBase), x))
    if all(is_SharedTensorBase):  # all x are SharedTensorBase
        if len(set(map(lambda u: u.module, x))) > 1:
            raise Exception("all x must have same module")
        module = x[0].module
        inner_value = tf.stack(list(map(lambda u: u.inner_value, x)), axis=axis)
        return SharedTensorBase(inner_value=inner_value, module=module)
    elif any(is_SharedTensorBase):
        raise Exception("must all x are SharedTensorBase or none of x is SharedTensorBase")

    if fixed_point is None:
        fixed_point = max(min(list(map(lambda u: u.fixedpoint, x))), StfConfig.default_fixed_point)
    x = list(map(lambda u: u.dup_with_precision(fixed_point), x))

    if all(list(map(lambda u: isinstance(u, PrivateTensorBase), x))):  # all x are PrivateTensorBase
        if len(set(map(lambda u: u.owner, x))) == 1:  # all x have same owner
            if len(set(map(lambda u: u.module, x))) > 1:
                raise Exception("all x must have same module")
            # fixed_point = min(list(map(lambda u: u.fixedpoint, x)))
            # x = list(map(lambda u: u.dup_with_precision(fixed_point), x))
            x_tf_tensor = list(map(lambda u: u.inner_value, x))
            y = tf.stack(x_tf_tensor, axis=axis)
            return PrivateTensorBase(owner=x[0].owner, fixedpoint=fixed_point, inner_value=y, module=x[0].module)
        else:
            owners = list(set(map(lambda u: u.owner, x)))
            if len(owners) > 2:
                raise StfCondException("len(owners)<=2", "owners={}".format(owners))

            x = list(
                map(
                    lambda u: u.to_SharedPairBase(
                        other_owner=owners[0] if owners[1] == u.owner else owners[1]),
                    x)
            )
            return stack(x, axis=axis, fixed_point=fixed_point)
    elif all(list(map(lambda u: isinstance(u, SharedPairBase), x))):  # all x[i] are SharedPairBase
        # y = x[0]
        # for i in range(1, len(x)):
        #     y = y.concat(x[i], axis=axis)
        # return y
        ownerL = x[0].ownerL
        ownerR = x[0].ownerR
        x = list(map(lambda u: u if u.ownerL == ownerL and u.ownerR == ownerR else u.mirror(), x))

        yL = stack(list(map(lambda u: u.xL, x)), axis=axis)
        yR = stack(list(map(lambda u: u.xR, x)), axis=axis)
        return SharedPairBase(ownerL=ownerL, ownerR=ownerR, xL=yL, xR=yR, fixedpoint=fixed_point)

    elif any(list(map(lambda u: isinstance(u, SharedPairBase), x))):  # exists x[i] is SharedPairBase
        for u in x:
            if isinstance(u, SharedPairBase):
                ownerL = u.ownerL
                ownerR = u.ownerR
                break
        for i in range(len(x)):
            if isinstance(x[i], PrivateTensorBase):
                if x[i].owner == ownerL:
                    other_owner = ownerR
                elif x[i].owner == ownerR:
                    other_owner = ownerL
                else:
                    raise Exception("must have x[i].owner == ownerL or x[i].owner == ownerR")
                x[i] = x[i].to_SharedPairBase(other_owner=other_owner)
        return stack(x, axis)
    else:

        raise Exception("x[i] must be PrivateTensorBase or SharedPairBase")
