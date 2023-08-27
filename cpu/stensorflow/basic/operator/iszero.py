#!/usr/bin/env python
# coding=utf-8
"""
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : iszero.py
   Author : Qizhi Zhang
   Email: zqz.math@gmail.ccom
   Create Time : 2022/12/25 下午1:00
   Description : description what the main function of this file
"""

from stensorflow.basic.basic_class.base import SharedPairBase, PrivateTensorBase
import tensorflow as tf
from typing import Tuple, Union
import numpy as np
from stensorflow.basic.protocol.equal import equal
import math
from stensorflow.exception.exception import StfDTypeException

def isZero(x: SharedPairBase):
    z = equal(x.xL, -x.xR, x.ownerL, x.ownerR)
    return z


def equalto(x: SharedPairBase, y: Union[SharedPairBase, PrivateTensorBase]):
    return isZero(x-y)
