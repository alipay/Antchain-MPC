#FROM python:3.4-alpine                 # << 基础镜像
#FROM continuumio/miniconda3
#FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morse-stf:v0.6
FROM reg.docker.alibaba-inc.com/morse/morsestfbase:v20200730_2


ADD . /morse-stf


WORKDIR /morse-stf
# << 设置工作目录

#RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#RUN pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
# RUN yum install wondershaper -y
RUN yum install net-tools -y
#RUN apt-get install vim -y
# RUN pip install coverage -i https://mirrors.aliyun.com/pypi/simple/
# RUN pip install xmlrunner -i https://mirrors.aliyun.com/pypi/simple/
# RUN pip install mock -i https://mirrors.aliyun.com/pypi/simple/
# RUN apt-get update
# RUN apt-get install curl -y
# RUN apt-get install gcc -y
#RUN wondershaper eth0 4096 4096
RUN export PYTHONPATH=$PYTHONPATH:/morse-stf
ENV PYTHONPATH /morse-stf
#安装依赖

#EXPOSE 8887 8888 8889
# CMD ["python", "app.py"]
#ENTRYPOINT ["/bin/bash"]
#<< 设置默认启动命令