
# Navigation for Secure Softmax/Sigmoid Artifact Installation

## Build

This project requires an NVIDIA GPU, and assumes you have your GPU driver already installed. 
We have built and run this project using Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. Download codes and checkout external modules
```shell
git submodule update --init --recursive
```
2. Prepare datasets
```shell
cd /{{path_to_gpu}}/gpu/scripts/
mkdir -p ../files/MNIST/
mkdir -p ../files/CIFAR10/
python3 download_mnist.py
python3 download_cifar.py
```
* require pytorch and torchvision to download dataset

3. Build a docker image

```shell
cd /{{path_to_gpu}}/gpu
docker build . -t piranha:1.0
```
* Dockerfile is provided in our codes.
* The repo and tag of the image can be customized.

4. Run a container and build Piranha

```shell
docker run -it --gpus all -v /{{path_to_gpu}}/gpu:/piranha piranha:1.0 
```

5. Run our scripts for experiments in the container

```shell
cd /piranha
nohup bash run.sh > log 2>&1 &
```
* The test logs are written into the file 'log'.


# Artifact Evaluation

This artifact includes the experiments provided in Table 5, 6, and 10. Running these experiments may cost about 20~30 minutes.
