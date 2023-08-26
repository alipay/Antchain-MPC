# Secure Softmax and Sigmoid
The branch of `sec_softmoid` contains Python implementation and C++ implementation of the paper entitled "Secure Softmax/Sigmoid for Machine-learning Computation".


Our open-source code is available at https://github.com/alipay/Antchain-MPC/tree/sec_softmoid. The artifact consists of CPU and GPU implementation prototypes in Python and C++, respectively. Besides, we provide scripts for reproducing experimental results in Section 6. Artifact evaluation is to reproduce results in replicating Tables 3-12 in the paper by executing the protocol-level and end-to-end training benchmarks. The evaluation consists of communication, running time, and training accuracy. As for GPU implementation in Piranha [33], participating parties require a machine equipped with a GPU and access to the NVIDIA CUDA toolkit.
The Python implementation (Apache license) follows the TenorFlow programming styles for providing user-oriented APIs. It supports both simulation by a local server and distributed training among three servers. In the Python prototype, we adopted the ideas of [9, 19, 25, 33] regarding our security model. In the offline phase, we realize a secure deterministic random bit generator (DRBG) conforming to CTR_DRBG standardized in NIST Special Publication 800-90A. The C++ implementation (MIT license) contains the plug-in blocks in the Piranha [33], known as a general-purpose solution for supporting later-inserting blocks. We added the blocks of Sigmoid protocol and Softmax protocol and truncation [19] by following Piranha’s programming interface.

## Artifact Checklists
- Code license and public available link.
- Python implementation.
- C++ implementation.
- Experimental scripts for Tables 3-12.
- README.md files for installation and running scripts.

**!!!!!!!!Unfinished and under construction**
