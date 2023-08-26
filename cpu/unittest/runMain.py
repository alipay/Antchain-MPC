# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2019-07-02 15:35
@file: runMain.py.py
@desc:
"""

# -*- coding: UTF-8 -*-
import os
import unittest

from xmlrunner import xmlrunner
from stensorflow.engine.start_server import start_local_server

if __name__ == '__main__':
    stf_home = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("stf_home=", stf_home)
    os.environ["stf_home"] = stf_home
    start_local_server(os.path.join(os.environ.get("stf_home", ".."), "conf", "config.json"))

    suite = unittest.TestSuite()
    # 找到目录下所有的以_test结尾的py文件
    # all_cases = unittest.defaultTestLoader.discover('.', '*_test.py')
    all_cases = unittest.defaultTestLoader.discover('unittest', 'test_*.py')
    #all_cases = unittest.defaultTestLoader.discover('.', 'test_*.py')
    for case in all_cases:
        
        # 把所有的测试用例添加进来
        suite.addTests(case)

    runner = xmlrunner.XMLTestRunner(output='report')

    runner.run(suite)
