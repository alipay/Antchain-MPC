g++ main.cpp -o main1 -L./ -I./ -lmorsebaselib_arm_linux -O3

g++ main.cpp libmorsebaselib_arm_linux.a -o main2 -L./ -I./ -O3
