#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : split
   Author : qizhi.zqz
   Email: qizhi.zqz@alibaba-inc.com
   Create Time : 2022/3/15 下午4:10
   Description : description what the main function of this file
"""


file = "/Users/qizhi.zqz/Downloads/epsilon_normalized.t"
file_x = "/Users/qizhi.zqz/Downloads/epsilon_normalized_test_x"
file_y = "/Users/qizhi.zqz/Downloads/epsilon_normalized_test_y"

with open(file, "r") as f, open(file_x, "w") as fx, open(file_y, "w") as fy:
    for _ in range(10):
        aline = f.readline()
        aline = aline.split(" ")
        if aline[0] == "1":
            label = "1\n"
        else:
            label = "0\n"
        features = list(map(lambda a: a.split(":")[1], aline[1:]))
        features = ",".join(features)
        fx.write(features)
        fy.write(label)



