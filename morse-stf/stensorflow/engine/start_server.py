#!/usr/bin/env python
# coding=utf-8
"""
   Ant Group
   Copyright (c) 2004-2020 All Rights Reserved.
   ------------------------------------------------------
   File Name : start_server
   Author : Qizhi Zhang
   Email: qizhi.zqz@antgroup.com
   Create Time : 2020-05-15 10:10
   Description : description what the main function of this file
"""
import tensorflow as tf
from stensorflow.global_var import StfConfig
import argparse

"""
cmd line:
export PYTHONPATH={your path}/morse-stf
"""

tf.compat.v1.disable_eager_execution()


def start_local_server(config_file):
    if StfConfig.serverL is None:
        StfConfig.load_config(config_file=config_file)
        if StfConfig.parties != 2 and StfConfig.parties != 3:
            raise Exception("must have StfConfig.parties == 2 or StfConfig.parties == 3")
        if StfConfig.parties == 3:
            cluster = tf.train.ClusterSpec({"workerL": StfConfig.workerL_hosts, "workerR": StfConfig.workerR_hosts, "RS": StfConfig.RS_hosts})
        else:
            cluster = tf.train.ClusterSpec({"workerL": StfConfig.workerL_hosts, "workerR": StfConfig.workerR_hosts})
        StfConfig.serverL = tf.distribute.Server(cluster, job_name="workerL", task_index=0)
        StfConfig.serverR = tf.distribute.Server(cluster, job_name="workerR", task_index=0)
        if StfConfig.parties == 3:
            StfConfig.serverRS = tf.distribute.Server(cluster, job_name="RS", task_index=0)



def start_distributed_server(config_file, job_name):
    StfConfig.load_config(config_file=config_file, job_name=job_name)
    if StfConfig.parties != 2 and StfConfig.parties != 3:
        raise Exception("must have StfConfig.parties == 2 or StfConfig.parties == 2")
    if StfConfig.parties == 3:
        cluster = tf.train.ClusterSpec({"workerL": StfConfig.workerL_hosts, "workerR": StfConfig.workerR_hosts, "RS": StfConfig.RS_hosts})
    else:
        cluster = tf.train.ClusterSpec({"workerL": StfConfig.workerL_hosts, "workerR": StfConfig.workerR_hosts})
    server = tf.distribute.Server(cluster, job_name=job_name, task_index=0)
    server.join()

def start_client(config_file, job_name):
    StfConfig.load_config(config_file=config_file, job_name=job_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--player', type=str, choices=["workerL", "workerR", "RS"], required=True)
    parser.add_argument('--config_file', type=str, default="./conf/config.json")
    args = parser.parse_args()
    job_name = args.player
    config_file = args.config_file

    start_distributed_server(config_file, job_name)


if __name__ == '__main__':
    main()



