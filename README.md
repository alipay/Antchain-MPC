# Antchain-MPC

Antchain-MPC is a open-source library for providing multiparty computation (MPC) algorithms/protocols. It includes Morse-STF, ...


## Key Branches
`sec_softmoid`: Secure Softmax/Sigmoid for Machine-learning Computation. [GitHub Link](https://github.com/alipay/Antchain-MPC/tree/sec_softmoid)

Others: Coming later.

## News

**Sep 3, 2022**

🎉🎉🎉  <font color=#DC143C> [MPC-Secure KMeans](https://github.com/alipay/Antchain-MPC/blob/main/morse-stf/stensorflow/ml/secure_k_means.py), [Partially-Secure PCA](https://github.com/alipay/Antchain-MPC/blob/main/morse-stf/stensorflow/ml/partially_secure_pca.py), [Fully-Secure PCA](https://github.com/alipay/Antchain-MPC/blob/main/morse-stf/stensorflow/ml/fully_secure_pca.py) have been updated. Running examples can be seen [here](https://github.com/alipay/Antchain-MPC/tree/main/morse-stf/examples).
 Please note that fake inverse_sqrt (used in Fully-Secure PCA) is NOT secure. We will release the real code later due to business reasons. </font>

**June 13, 2022** 

<font> [User Guide](morse-stf/documents/user_guide.md) for Morse-STF has been updated. </font>

## Morse-STF

Morse-STF is a tool for privacy-preserving machine learning using MPC. We outline more information below:
| Folder      | Description |
| -----------  | ----------- |
| conf        | configuration files   |
| ducuments   |  descriptive files (keep updating)   |
| examples    | runnning examples for stensorflow |
| stensorflow | importing folder for all functionality |
| unitest| testing files for different units used in stensorflow |
| output| store output files |

[User Guide](morse-stf/documents/user_guide.md)

[Morse-STF快速开始](https://github.com/alipay/Antchain-MPC/wiki/MORSE-STF%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)

[Morse-STF使用手册](https://github.com/alipay/Antchain-MPC/wiki/MORSE-STF%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C)

### Python Package Index

If you do not prefer to git clone our repository, python packages are [here](https://pypi.org/project/morse-stf/) via [pip install](https://pip.pypa.io/en/stable/cli/pip_install/). 

### Disclaimer
Part of code has been used in [AntChain](https://www.antchain.net/home) products. If you use it for business, please don't hesitate to get in touch with our team.

For academic usage, please cite our paper (under anonymous review, to be updated).
