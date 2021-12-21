#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : global
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-14 14:30
   Description : description what the main function of this file
"""

import tensorflow as tf
import json
import platform
import os


class StfConfig:
    workerL_hosts = None
    workerR_hosts = None
    RS_hosts = None
    workerL = None
    workerR = None
    RS = None
    serverL = None
    serverR = None
    serverRS = None
    default_fixed_point = 14
    prf_flag = True
    compress_flag = True
    target = None
    train_file_onL = None
    train_file_onR = None
    pred_file_onL = None
    pred_file_onR = None
    predict_to_file = None
    custom_ops_path = None
    random_module = None
    conv_module = None
    pool_module = None
    homo_module = None
    upper_bd_int64 = (1 << 63) - 1
    lower_bd_int64 = -(1 << 63)
    to_str_precision = 6
    stf_home = None
    stf_home_workerL = None
    stf_home_workerR = None
    stf_home_RS = None
    coll_name_vars_random = "var_for_random"  # collection name for vars for random
    coll_name_vars_homo = "var_for_homo"  # collection name for vars for homo
    drelu = "log"
    invert_iter_num = 32
    softmax_iter_num = 32
    parties = 3
    pre_produce_flag = None
    offline_model = None
    pre_produce_list = []
    offline_triple_multiplex = 10000

    @classmethod
    def load_config(cls, config_file, job_name=None):
        with open(config_file) as f:
            config_dict = json.load(f)
        StfConfig.parties = config_dict.get("parties")
        if StfConfig.parties is None:
            StfConfig.parties = 3
        hosts_dict = config_dict.get("hosts")
        workerL_ip_port = hosts_dict.get("workerL")
        workerR_ip_port = hosts_dict.get("workerR")
        if StfConfig.parties == 3:
            RS_ip_port = hosts_dict.get("RS")


        StfConfig.workerL_hosts = workerL_ip_port.split(",")
        StfConfig.workerR_hosts = workerR_ip_port.split(",")
        if StfConfig.parties == 3:
            StfConfig.RS_hosts = RS_ip_port.split(",")

        workerL = map(lambda task_id: tf.DeviceSpec(job="workerL", task=task_id), range(len(StfConfig.workerL_hosts)))
        StfConfig.workerL = list(workerL)
        workerR = map(lambda task_id: tf.DeviceSpec(job="workerR", task=task_id), range(len(StfConfig.workerR_hosts)))
        StfConfig.workerR = list(workerR)
        if StfConfig.parties == 3:
            RS = map(lambda task_id: tf.DeviceSpec(job="RS", task=task_id), range(len(StfConfig.RS_hosts)))
            StfConfig.RS = list(RS)

        if StfConfig.parties == 2:
            StfConfig.pre_produce_flag = config_dict.get("pre_produce_flag")
            StfConfig.offline_model = config_dict.get("offline_model")



        if job_name:
            sess_worker = job_name
        else:
            sess_worker = "workerR"
        #sess_worker = config_dict.get("sess_worker")
        if sess_worker == "workerL":
            sess_ip_port = workerL_ip_port
        elif sess_worker == "workerR":
            sess_ip_port = workerR_ip_port
        elif sess_worker == "RS":
            sess_ip_port = RS_ip_port
        else:
            raise Exception("must have sess_worker==workerL or sess_worker==workerR "
                            "or sess_worker==RS, "
                            "but sess_worker={}".format(sess_worker))
        StfConfig.target = "grpc://" + sess_ip_port

        StfConfig.prf_flag = config_dict.get("prf_flag")
        StfConfig.compress_flag = config_dict.get("compress_flag")
        StfConfig.default_fixed_point = config_dict.get("default_fixed_point")
        if job_name is None:
            stf_home = os.environ.get('stf_home')
            if stf_home is None:
                stf_home = config_dict.get("stf_home")
            StfConfig.stf_home = stf_home
            StfConfig.stf_home_workerL = stf_home
            StfConfig.stf_home_workerR = stf_home
            if StfConfig.parties == 3:
                StfConfig.stf_home_RS = stf_home
        else:
            StfConfig.stf_home_workerL = config_dict.get("stf_home_workerL")
            StfConfig.stf_home_workerR = config_dict.get("stf_home_workerR")
            if StfConfig.parties == 3:
                StfConfig.stf_home_RS = config_dict.get("stf_home_RS")
            if job_name == "workerL":
                StfConfig.stf_home = StfConfig.stf_home_workerL
            elif job_name == "workerR":
                StfConfig.stf_home = StfConfig.stf_home_workerR
            elif job_name == "RS":
                StfConfig.stf_home = StfConfig.stf_home_RS
            else:
                raise Exception("must have job_name in {'workerL', 'workerR', 'rs'}.")

        # -------ml config ---------------------
        ml_config = config_dict.get("ml")
        if ml_config is not None:
            train_cfg = ml_config.get("dataset_train")
            pred_cfg = ml_config.get("dataset_predict")
            StfConfig.train_file_onL = os.path.join(StfConfig.stf_home_workerL, train_cfg.get("L"))
            StfConfig.train_file_onR = os.path.join(StfConfig.stf_home_workerR, train_cfg.get("R"))
            StfConfig.pred_file_onL = os.path.join(StfConfig.stf_home_workerL, pred_cfg.get("L"))
            StfConfig.pred_file_onR = os.path.join(StfConfig.stf_home_workerR, pred_cfg.get("R"))
            if sess_worker == "workerL":
                StfConfig.predict_to_file = os.path.join(StfConfig.stf_home_workerL, ml_config.get("predict_to_file"))
            elif sess_worker == "workerR":
                StfConfig.predict_to_file = os.path.join(StfConfig.stf_home_workerR, ml_config.get("predict_to_file"))
            else:
                raise Exception("must have sess_worker==workerL or sess_worker==workerR, "
                                "but sess_worker={}".format(sess_worker))

        #StfConfig.cops_path = os.path.join(StfConfig.stf_home, config_dict.get("cops_path"))

        StfConfig.cops_path = os.environ['HOME'] + '/stf_cops'
        operating_system_platform = platform.system()

        if operating_system_platform == "Linux":
            StfConfig.random_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_random_linux.so'))
            StfConfig.pool_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_int64pooling_linux.so'))
            StfConfig.conv_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_int64conv2d_linux.so'))
            StfConfig.homo_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, "_stf_homo_linux.so"))

        elif operating_system_platform == "Darwin":
            StfConfig.random_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_random_macos.so'))
            StfConfig.pool_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_int64pooling_macos.so'))
            StfConfig.conv_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, '_stf_int64conv2d_macos.so'))
            StfConfig.homo_module = tf.load_op_library(
                os.path.join(StfConfig.cops_path, "_stf_homo_macos.so"))
        else:
            raise Exception("only support Linux or macos.")

        # # ----------- protocols ------------
        protocols = config_dict.get("protocols")
        StfConfig.drelu = protocols.get("drelu")
