TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


# all
RBG_CPPS=$(find src -type f)
echo $RBG_CPPS
SUFFIX="-undefined dynamic_lookup"
plat=`uname`
name_suffix="macos"
echo $plat
if [[ "$plat" == "Linux"* ]]; then
  SUFFIX=""
  name_suffix="linux"
fi

g++ -std=c++11  -shared stf_random.cc $RBG_CPPS -Iinc  -o _stf_random_${name_suffix}.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $SUFFIX -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11  -shared stf_conv2d.cc $RBG_CPPS -Iinc  -o _stf_int64conv2d_${name_suffix}.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $SUFFIX -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11  -shared stf_pooling.cc $RBG_CPPS -Iinc  -o _stf_int64pooling_${name_suffix}.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $SUFFIX -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11  -shared _stf_homo.cc libmorsehemt_${name_suffix}.a $RBG_CPPS -Iinc -o _stf_homo_${name_suffix}.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $SUFFIX -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11  -shared stf_test.cc $RBG_CPPS -Iinc  -o _stf_test_${name_suffix}.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $SUFFIX -D_GLIBCXX_USE_CXX11_ABI=0
