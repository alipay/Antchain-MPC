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
file_xx = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_xx"
file_xy = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_xy"
file_x30 = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x30"
file_x31 = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x31"
file_x32y = "/Users/qizhi.zqz/Documents/dataset/epsilon_normalized_test_x32y"

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

def split2():
    """
    split to 2 part
    :return:
    """
    with open(file_x) as fx, open(file_y) as fy, open(file_xx, "w") as fxx, open(file_xy, "w") as fxy:
        aline_xx = ",".join(("x{i}".format(i=i) for i in range(1500)))+"\n"
        aline_xy = ",".join(("x{i}".format(i=i) for i in range(1500, 3000)))+",y\n"
        fxx.write(aline_xx)
        fxy.write(aline_xy)
        while True:
            aline_x = fx.readline()
            if aline_x is None:
                break
            if len(aline_x) <= 2:
                break
            aline_y = fy.readline()
            if aline_y is None:
                break
            if len(aline_y) == 0:
                break

            aline_x = aline_x.strip()
            aline_y = aline_y.strip()


            aline_x = aline_x.split(",")

            aline_xx = aline_x[0:1500]
            aline_xy = aline_x[1500:3000]+[aline_y]

            aline_xx = ",".join(aline_xx)+"\n"
            aline_xy = ",".join(aline_xy)+"\n"
            fxx.write(aline_xx)
            fxy.write(aline_xy)

def split3():
    """
    split to 3 part
    :return:
    """
    with open(file_x) as fx, open(file_y) as fy, open(file_x30, "w") as fx30, \
            open(file_x31, "w") as fx31, open(file_x32y, "w") as fx32y:

        aline_x30 = ",".join(("x{i}".format(i=i) for i in range(1000)))+"\n"
        aline_x31 = ",".join(("x{i}".format(i=i) for i in range(1000, 2000)))+"\n"
        aline_x32y = ",".join(("x{i}".format(i=i) for i in range(2000, 3000)))+",y\n"
        fx30.write(aline_x30)
        fx31.write(aline_x31)
        fx32y.write(aline_x32y)
        while True:
            aline_x = fx.readline()
            if aline_x is None:
                break
            if len(aline_x) <= 2:
                break
            aline_y = fy.readline()
            if aline_y is None:
                break
            if len(aline_y) == 0:
                break

            aline_x = aline_x.strip()
            aline_y = aline_y.strip()


            aline_x = aline_x.split(",")

            aline_x30 = aline_x[0:1000]
            aline_x31 = aline_x[1000:2000]
            aline_x32y = aline_x[2000:3000]+[aline_y]

            aline_x30 = ",".join(aline_x30)+"\n"
            aline_x31 = ",".join(aline_x31) + "\n"
            aline_x32y = ",".join(aline_x32y) + "\n"

            fx30.write(aline_x30)
            fx31.write(aline_x31)
            fx32y.write(aline_x32y)




if __name__ == '__main__':
    #get_dataset2()
    split2()
    # split3()