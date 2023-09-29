#!/bin/bash

build(){
  echo "building"
  make -j PIRANHA_FLAGS="-DFLOAT_PRECISION=16 -DTWOPC" 2>&1 | grep -i error
  echo "built"
}
kill_piranha(){
  ps aux|grep -v "grep"|grep "piranha"|awk -F " " '{print $2}'|xargs -I{} kill {}
}

kill_piranha
mkdir -p /piranha/build

chmod +x ./files/samples/localhost_runner.sh

echo "==== ==== ==== ==== ==== ==== Table 5 (r = 16) ==== ==== ==== ==== ==== ===="
sed -i 's/"network":.*$/\"network":\ "files\/models\/secureml-norelu.json",/' files/samples/localhost_config.json

sed -i 's/"run_unit_tests": false,/"run_unit_tests": true,/g' files/samples/localhost_config.json
sed -i 's/"unit_test_only": false,/"unit_test_only": true,/g' files/samples/localhost_config.json

sed -i 's/^#define SOFTMAX_ITER_NUM 32/#define SOFTMAX_ITER_NUM 16/g' src/mpc/TPC.inl
sed -i 's/^\/\/#define TEST_SOFTMAX true/#define TEST_SOFTMAX true/g' src/mpc/TPC.inl
sed -i 's/^#define TEST_SIGMOID true/\/\/#define TEST_SIGMOID true/g' src/mpc/TPC.inl
build
./files/samples/localhost_runner.sh 2>&1
#| grep  "\[ RUN      \] FuncTest/0.Softmax\|\[ RUN      \] FuncTest/0.Sin2kpix"  -A 17 |
kill_piranha
echo "==== ==== ==== ==== ==== ==== Table 5 (r = 32) ==== ==== ==== ==== ==== ===="
sed -i 's/^#define SOFTMAX_ITER_NUM 16/#define SOFTMAX_ITER_NUM 32/g' src/mpc/TPC.inl
build
./files/samples/localhost_runner.sh 2>&1
kill_piranha


#
echo "==== ==== ==== ==== ==== ==== Table 6 ==== ==== ==== ==== ==== ===="
sed -i 's/^#define SOFTMAX_ITER_NUM 32/#define SOFTMAX_ITER_NUM 16/g' src/mpc/TPC.inl
sed -i 's/^#define TEST_SOFTMAX true/\/\/#define TEST_SOFTMAX true/g' src/mpc/TPC.inl
sed -i 's/^\/\/#define TEST_SIGMOID true/#define TEST_SIGMOID true/g' src/mpc/TPC.inl
build
./files/samples/localhost_runner.sh 2>&1 #|grep "\[ RUN      \] FuncTest/0.Sigmoid" -A 17
kill_piranha



echo "==== ==== ==== ==== ==== ==== Table 10 ==== ==== ==== ==== ==== ===="
sed -i 's/"run_unit_tests": true,/"run_unit_tests": false,/g' files/samples/localhost_config.json
sed -i 's/"unit_test_only": true,/"unit_test_only": false,/g' files/samples/localhost_config.json

echo "==== ==== LeNet (MNIST) ==== ===="
sed -i 's/"network":.*$/\"network":\ "files\/models\/lenet-norelu.json",/' files/samples/localhost_config.json
build
./files/samples/localhost_runner.sh 2>&1
kill_piranha

echo "==== ==== AlexNet (CIFAR10) ==== ===="
sed -i 's/"network":.*$/\"network":\ "files\/models\/alexnet-cifar10-norelu.json",/' files/samples/localhost_config.json
build
./files/samples/localhost_runner.sh 2>&1
kill_piranha

echo "==== ==== VGG16 (CIFAR10) ==== ===="
sed -i 's/"network".*$/\"network":\ "files\/models\/vgg16-cifar10-norelu.json",/' files/samples/localhost_config.json
build
./files/samples/localhost_runner.sh 2>&1
kill_piranha

echo "==== ==== ResNet18 (CIFAR10) ==== ===="
sed -i 's/"network":.*$/\"network":\ "files\/models\/resnet18-cifar10-norelu.json",/' files/samples/localhost_config.json
build
./files/samples/localhost_runner.sh 2>&1
kill_piranha

