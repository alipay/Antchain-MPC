# Secure Softmax and Sigmoid
The branch of `sec_softmoid` contains Python and C++ implementations of the paper 

**Secure Softmax/Sigmoid for Machine-learning Computation. (ACSAC'23)**

**by Yu Zheng#, Qizhi Zhang#, Sherman S.M. Chow, Yuxiang Peng, Sijun Tan, Lichun Li, and Shan Yin**. (# denotes equal contribution)

This artifact has been awarded  **Artifact Functional Badge**, **Artifact Reusable Badge**, and **Results Reproduced Badge**.

Softmax and sigmoid, composing exponential functions $e^x$ with division $(1/𝑥)$, are activation functions often required in training. Secure computation on non-linear, unbounded $1/𝑥$ and $𝑒^𝑥$ is already challenging, let alone their composition. Prior works aim to compute softmax by its exact formula via iteration (CrypTen, NeurIPS ’21) or with ASM approximation (Falcon, PoPETS ’21). They fall short in efficiency and/or accuracy. For sigmoid, existing solutions such as ABY 2.0 (Usenix Security ’21) compute it via piecewise functions, incurring logarithmic communication rounds.

We study a rarely-explored approach to secure computation using ordinary differential equations and Fourier series for numerical approximation of rational/trigonometric polynomials over composition rings. Our results include 
- 1) the first constant-round protocol for softmax;
- 2) the first 1-round error-bounded protocol for approximating sigmoid.
They reduce communication by ∼83% and ∼95%, respectively, when compared with prior arts, shortening the private training process with much less communication than state-of-the-art frameworks, namely, CryptGPU (S&P ’21), Piranha (Usenix Security ’22), and quantized training from MP-SPDZ (ICML ’22), while maintaining competitive accuracy.


## Abstract for Artifact Evaluation 
Our open-source code is available at [https://github.com/alipay/Antchain-MPC/tree/sec_softmoid](https://github.com/alipay/Antchain-MPC/tree/sec_softmoid). The artifact consists of CPU and GPU implementation prototypes in Python and C++, respectively. Besides, we provide scripts for reproducing experimental results in Section 7. Artifact evaluation is to reproduce the results of replicating Tables 3-12 in the paper by executing the protocol-level and end-to-end training benchmarks. The evaluation consists of communication, running time, and training accuracy. As for GPU implementation in Piranha (Usenix Security ’22), participating parties require a machine equipped with a GPU and access to the NVIDIA CUDA toolkit.
The Python implementation (Apache license) follows the TensorFlow programming styles for providing user-oriented APIs. It supports both simulation by a local server and distributed training among three servers. In the Python prototype, we adopted the ideas of Cheetah (Usenix Security'22), SecureML (S\&P' 17), CrypTFlow2 (CCS'20), Piranha (Usenix Security ’22) regarding our security model. In the offline phase, we realize a secure deterministic random bit generator (DRBG) conforming to CTR_DRBG standardized in NIST Special Publication 800-90A. The C++ implementation (MIT license) contains the plug-in blocks in the Piranha, known as a general-purpose solution for supporting later inserting blocks. We added the blocks of Sigmoid protocol and Softmax protocol and truncation by following Piranha’s programming interface.

## Artifact Checklist
- Code license and publicly available link.
- Python implementation.
- C++ implementation.
- Experimental scripts for Tables 3-12.
- `README.md` files for installation and running scripts.

## Download the Code

`git clone -b sec_softmoid https://github.com/alipay/Antchain-MPC.git`

`cd Antchain-MPC`

`git checkout -b sec_softmoid origin/sec_softmoid`

`mv gpu piranha`

`cd cpu` or `cd piranha` 

## Evaluation and Experimental Scripts

### Preparation and Description
- Program: Python and C++ languages, Docker, TensorFlow 2, and CUDA 11.6.
- Metric: communication, training time, and model accuracy. Datasets: MNIST, CIFAR-10.
- Outputs: Results replicate Tables 3-12. Running time would be different on different servers.
- Hardware dependence: NVIDIA GPUs are required for microbenchmarks and macro-benchmarks in Piranha.
- Time: It roughly takes 1~2 hours to set up and 2 hours to run most experiments (except for model accuracy). The time may vary depending on the power of the server.
- Server reference: We have tested our codes on three types of servers, including
  (1) Alibaba cloud servers equipped with 8-core 2.50GHz CPU of 64GB RAM and NVIDIA-T4 GPU of 16GB RAM,
  (2) the commodity server equipped with two 24-core 2.10GHz CPUs and two NVIDIA-A40 GPUs of 48GB RAM, 
  (3) MacBook Pro (CPU only).
- Public availability: The code is available at [https://github.com/alipay/Antchain-MPC/tree/sec_softmoid](https://github.com/alipay/Antchain-MPC/tree/sec_softmoid). Since the code repository is owned by Ant Group and not under the authors' management permanently, the authors maintain a mirror repository at [https://github.com/MathCrypt0/softmoid-public](https://github.com/MathCrypt0/softmoid-public).
- Code licenses: The Python implementation is under Apache license according to code-disclosure rules in Ant Group. The C++ implementation follows Piranha’s license – MIT license.

### Protocol-level Communication
In the `cpu` folder, run the script of `experiment_table_3&4.py` to obtain the experimental results of Tables 3,4.

### CPU-based Experiments 
Follow the `README.md` instruction in `cpu` folder for installation. Then, run the script `run.sh` to obtain the results of Tables 7,8,9,11,12. Since training for model accuracy costs a relatively long time (i.e., several hours), we put the relevant experiments at last. For user convenience, we attach the record of parameter tuning in the file ./cpu/artifacts/record.csv.

### GPU-based Experiments 
Follow the `README.md` instruction in `piranha` folder to install the C++ implementation. Then, download the `MNIST` and `CIFAR-10` datasets. At last, execute the script `run.sh` in the Docker container to get results of Tables 5,6,10.

## Implementation Details
### Python Implementation
To be added ...
### C++ Implementation 
To be added ...

## Acknowledgement
The authors wholeheartedly appreciate the invaluable feedback from the anonymous shepherd, reviewers, and artifacts evaluation committee. We thank Yuan Zhao, Yashun Zhou, Dong Yin, and Jiaofu Zhang at Ant Group for their insightful discussions and endeavors on coding, and Jiafan Wang for his help and guidance to Yu. Special thanks go to David Wu, Florian Kerschbaum, and Jean-Pierre Hubaux for their constructive suggestions for Yu’s poster at EPFL Summer Research Institute.  

## Disclaimer & Citation

Intellectual properties have been protected by Chinese patents and are free for academic usage. For business usage in the Chinese market, please get in touch with Morse team at Ant Group.

If you feel our work interesting, welcome to cite our work
```
@inproceedings{acsac/ZhengZCPTLY23,
  author       = {Yu Zheng and 
                  Qizhi Zhang and
                 Sherman S.M. Chow and
                 Yuxiang Peng and
                 Sijun Tan and
                 Lichun Li and
                 Shan Yin},
  title        = {Secure Softmax/Sigmoid for Machine-learning Computation},
  booktitle    = {ACSAC},
  year         = {2023}
}
```


