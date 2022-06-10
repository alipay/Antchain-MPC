# User Guide
[toc]


## Overview
AntChain-MPC is an open-source Library for secure computation over private data. It contains Morse-STF, ...


## Morse-STF
Morse-STF is a module for secure multi-party computation atop TensorFlow. It can protect user's inputs when computing the collaborative output.

![two-party](topology_2parties.png "secure two-party computation")

Morse-STF contains *secure* arithmetic computation, Boolean calculation, sequential operation, primitive operation, and many functions for machine learning.


### Install
See tutorials.


### Running Examples 

|  Operation             | Setting     |          Remark                    |
|  ---                   | ---         |    ---                             | 
| matrix multiplication  | l2,d2,l3,d3 | private inputs from two parties    |
| linear regression      | l2,d2,l3,d3 | private features from one party    |
| linear regression      | l2,d2,l3,d3 | private features from two parties  |
| DNN prediction         | l3,d3       | private features from two parties  |
| DNN prediction         | l3,d3       | private features from one party    |

l2 - local simulation for two-party computation; l3 - local simulation for three-party computation;   
d2 - distributed deployment for two-party computation; d3 - distributed deployment for three-party computation; 

### Data Types
Morse-STF defines three data types: `class PrivateTensor`, `class SharedTensor`, `class SharedPair`.
