# syntax=docker/dockerfile:1

# Build piranha binary
FROM pengyuxiang/cuda:11.6.0-devel-ubuntu20.04 AS builder
#ARG flags

WORKDIR /

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "" > /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt update
RUN apt install build-essential libssl-dev libgtest-dev cmake -y

RUN mkdir -p /usr/src/gtest/build
WORKDIR /usr/src/gtest/build
RUN cmake ..
RUN make
WORKDIR /piranha
CMD ["/bin/bash"]