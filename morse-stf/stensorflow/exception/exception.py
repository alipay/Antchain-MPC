#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2021 All Rights Reserved.
   ------------------------------------------------------
   File Name : exception
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/5/19 下午3:51
   Description : description what the main function of this file
"""
import warnings


class StfException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.__repr__()

    def __unicode__(self):
        return self.__repr__()


class StfNoneException(StfException):
    def __init__(self, obj_name):
        self.msg = "{} should not be None.".format(obj_name)


class StfTypeException(StfException):
    def __init__(self, obj_name, expect_type, real_type):
        self.msg = "{} should be {}, but is {}.".format(obj_name, expect_type, real_type)


class StfDTypeException(StfException):
    def __init__(self, obj_name, expect_dtypes, real_dtype):
        self.msg = "{}.dtype should in {}, but is {}.".format(obj_name, expect_dtypes, real_dtype)


class StfValueException(StfException):
    def __init__(self, var_name, expect_value, real_value):
        self.msg = "{} should equal to {}, but in fact equal to {}.".format(var_name, expect_value, real_value)


class StfEqualException(StfException):

    def __init__(self, var1_name, var2_name, var1, var2):
        self.msg = "should have {}={}, but in fact {}={}, {}={}".format(var1_name, var2_name, var1_name,
                                                                        var1, var2_name, var2)


class StfEqualWarning:
    def __init__(self, var1_name, var2_name, var1, var2):
        self.msg = "should have {}={}, but in fact {}={}, {}={}".format(var1_name, var2_name, var1_name,
                                                                        var1, var2_name, var2)
        warnings.warn(self.msg)


class StfCondException(StfException):
    def __init__(self, cond, real):
        self.msg = "must have {}, but in fact {}.".format(cond, real)


def check_is_not_None(**kwargs):
    dict_name_value = dict(kwargs)
    for name, value in dict_name_value.items():
        if value is None:
            raise StfNoneException(obj_name=name)
