g++ -fPIC -shared *.cpp -O3 -I./ -o libmorsebaselib_arm_linux.so

g++ -fPIC -c *.cpp
ar rcs libmorsebaselib_arm_linux.a *.o
rm *.o

