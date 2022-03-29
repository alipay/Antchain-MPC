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
import numpy as np
params_ext1000 = []
for n in range(1000):
    t = pow(5, n, (1<<13))
    params_ext1000.append(t/(1<<13))
print(params_ext1000)

file = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized.t"
file_x = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x"
file_y = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_y"

def get_dataset():

    with open(file, "r") as f, open(file_x, "w") as fx, open(file_y, "w") as fy:
        while True:
            aline = f.readline()
            if aline is None:
                break
            if len(aline)<=2:
                break
            aline = aline[:-1]
            aline = aline.split(" ")
            if aline[0] == "1":
                label = "1\n"
            else:
                label = "0\n"
            features = list(map(lambda a: a.split(":")[1], aline[1:]))
            ext = np.random.normal(size=[1000])
            y = np.sum(ext * params_ext1000)
            if y * int(aline[0]) < 0:
                ext = -ext
            # ext = ext.astype("str")
            features = features + ["%.6f"%x for x in ext]
            print(features[0:10])
            features = ",".join(features)+"\n"
            print(features[0:1000])
            fx.write(features)
            fy.write(label)


def get_dataset2():
    """
    在上个版本的基础上，对每条记录增加一条加了噪声的记录
    :return:
    """
    with open(file, "r") as f, open(file_x, "w") as fx, open(file_y, "w") as fy:
        while True:
            aline = f.readline()
            if aline is None:
                break
            if len(aline) <= 2:
                break
            aline = aline[:-1]
            aline = aline.split(" ")
            if aline[0] == "1":
                label = "1\n"
            else:
                label = "0\n"
            features = list(map(lambda a: a.split(":")[1], aline[1:]))
            ext = np.random.normal(size=[1000])
            y = np.sum(ext * params_ext1000)
            if y * int(aline[0]) < 0:
                ext = -ext
            # ext = ext.astype("str")
            features = features + ["%.6f" % x for x in ext]
            features_bak = features
            print(features[0:10])
            features = ",".join(features) + "\n"
            print(features[0:1000])
            fx.write(features)
            fy.write(label)
            features = np.array(features_bak).astype("float")
            features = features + np.random.normal(size=[3000])
            print("len(feature)=", len(features))
            features = ",".join(["%.6f" % x for x in features])+"\n"
            if (int(label == "1\n") + int(np.random.uniform(size=1) > 0.1)) % 2 == 0:
                label = "1\n"
            else:
                label = "0\n"

            fx.write(features)
            fy.write(label)



if __name__ == '__main__':
    get_dataset2()