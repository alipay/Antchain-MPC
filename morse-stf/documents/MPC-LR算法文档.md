#MPC-LR算法文档

##2方训练

###input: 
L方持有xL\_train: [batch\_size, num\_features\_L], 

R方持有xR\_train: : [batch\_size, num\_features\_R] 和 y\_train: : [batch\_size, 1]
###output: R方拿到模型参数

Step 0. L，R双方随机初始化share形式的w\_L : [num\_features\_L, 1], w\_R:  [num\_features\_R, 1], b: [1]

Step 1 (前向运算图). L, R根据安全乘法协议和安全加法协议构造运算图，运算图的输出为

$$
y_{hat} = \mbox{sigmoid}(\mbox{out})  
$$ 

其中
$$
out = x_L  w_L + x_R  w_R + b
$$



Step 2 (反向运算图). L, R根据安全加法协议和乘法协议构造运算图用以计算
$$
\frac{\partial loss}{\partial \mbox{out}} = y_{hat} - y
$$
以及
$$
\frac{\partial loss}{\partial w_L} = \frac{1}{batch\_size} x_L ^T \frac{\partial loss}{\partial \mbox{out}} \\
\frac{\partial loss}{\partial w_R} = \frac{1}{batch\_size} x_R ^T \frac{\partial loss}{\partial \mbox{out}}   \\
\frac{\partial loss}{\partial b} = \mbox{mean}(\frac{\partial loss}{\partial \mbox{out}})
$$

Step 3  L, R根据安全加法协议和明文乘shares协议构造运算图用以更新
$$
w_L \leftarrow w_L - learning\_rate * \frac{\partial loss}{\partial w_L} \\
w_R \leftarrow  w_R - learning\_rate * \frac{\partial loss}{\partial w_R} \\
b \leftarrow  b - learning\_rate * \frac{\partial loss}{\partial b} 
$$

Step 4. L，R双方运行计算图batch_num次

Step 5.  L方将持有的$w_L, w_R, b$的share发送给R方，由R方合并出明文。

##2方预测

### input
L方持有xL\_test: [batch\_size, num\_features\_L], 

R方持有xR\_test: : [batch\_size, num\_features\_R],  $w_L, w_R, b$

###output
R方得到预测结果

Step 0. R方将$w_L$发送给L方

Step 1 在L方构建计算图用以计算$u_L:=xL\_{test} @ w_L$, 并发送给R方

Step 2 在R方构建计算图用以计算$y_{hat}:=\mbox{sigmoid}(u_L + xR \_{test} @ w_R + b)$

Step 3. L, R双方运行计算图pred\_batch\_num次, R方返回  $y_{hat}$

##多方训练



### input

$P_i$方持有xi\_train: [batch\_size, num\_features\_i], 其中 $i=0, \cdots, n-1$, 且$P_0=L$, $P_1=R$.

###output: 
R方得到模型参数

Step 0.  对于$i>=2$，$P_i$方将数据拆share, 分别发送给$P_0$方和$P_1$方

Step 1. P0, P1双方随机初始化share形式的w\_i : [num\_features\_i, 1], for $i \in \{0, 1, \cdots, n-1\}$  以及 b: [1]

Step 2.(前向运算图) P0, P1根据安全乘法协议和安全加法协议构造运算图，运算图的输出为
$$
y_{hat} = \mbox{sigmoid}(\mbox{out})
$$
其中
$$
out = \sum _i x_i  w_i + b
$$

Step 3. (反向运算图). P0, P1根据安全加法协议和乘法协议构造运算图用以计算
$$
\frac{\partial loss}{\partial \mbox{out}} = y_{hat} - y
$$
以及
$$
\frac{\partial loss}{\partial w_i} = \frac{1}{batch\_size} x_i ^T \frac{\partial loss}{\partial \mbox{out}} \\
\mbox{ for } i=0, \cdots , n-1 \\
\frac{\partial loss}{\partial b} = \mbox{mean}(\frac{\partial loss}{\partial \mbox{out}})
$$

Step 4. P0, P1根据安全加法协议和明文乘shares协议构造运算图用以更新
$$
w_i \leftarrow w_i - learning\_rate * \frac{\partial loss}{\partial w_i} \\
\mbox{ for } i=0, \cdots , n-1\\
b \leftarrow  b - learning\_rate * \frac{\partial loss}{\partial b} 
$$

Step 5. P0，P1双方运行计算图batch_num次

Step 6.  P1方将持有的$w_i, b$的share发送给P0方，由P0方合并出明文。


##多方预测

### input:

$P_i$方持有xi\_test: [batch\_size, num\_features\_i], 其中 $i=0, \cdots, n-1$. $P_0$方还另外持有$w_i$, $b$

### output:

$P_0$方得到预测结果

Step 0. $P_0$方将$w_i$发送给$P_i$方

Step 1 每个$P_i$方自行计算$u_i:=xi\_{test} @ w_i$, 并发送给$P_0$方

Step 2 $P_0$方计算$y_{hat}:=\mbox{sigmoid}( \sum _i u_i + b)$



##双方DNN训练

###input: 

L方持有xL\_train: [batch\_size, num\_features\_L], 

R方持有xR\_train: : [batch\_size, num\_features\_R] 和 y\_train: : [batch\_size, 1]

###output: 

R方拿到模型参数

Step 0a. L方将第1层参数中与自己维数匹配的部分初始化为自己的私有变量$W^L _1$: [num\_features\_L, dim1], R方将第1层参数中与自己维数匹配的部分初始化为自己的私有变量$W^R _1$:  [num\_features\_R, dim1], 并将第一层参数中的常数项初始化为自己的私有变量$b_1$:  [dim1]; 

Step 0b. L方，R方将第一层参数及常数项初始化为share方式存储于L, R双方的变量$w_2$: [dim1, dim2], $b_2$: [dim2]

Step 0c. R方自己将后面各层的参数初始化为自己的私有变量

Step 1. (前向运算图第一层) L R双方方构造运算图以计算 
$$
u_1:= x_L w^L _1 + x_R w^R _1 +b_1
$$
其中的矩阵乘法分别由L, R双方各自本地运算，得到的结果以share方式存储于L, R双方

Step 2. (前向运算图第二层) L  R双方构建运算图以计算
$$
x_2:=ReLU(u_1)
$$
以及 
$u_2:=x_2 w_2 +b_2$
其中的运算均以share形式进行 

并将计算 结果$u_2$交给R方。

Step 3.  (前向运算图后续层)   R方自己 构建运算图以计算神经网络的后续各层的前向传播，直至输出预测结果$y$

Step 4.  (反向运算图后续层) R方自己构建运算图以计算神经网络的反向传播的后面几层，以计算出
$$
\frac{\partial {loss}}{\partial w_i}, \frac{\partial {loss}}{\partial b_i}  \mbox{ for } i>=3
$$
以及
$$
\frac{\partial {loss}}{\partial u_2}
$$

Step 5 (反向传播第2层) L, R双方构建计算图以计算神经网络反向传播的第2层，即
$$
\frac{\partial {loss}}{\partial w_2}:= \frac{1}{batch\_size} x_2 ^T \frac{\partial loss}{\partial u_2}
$$
$$
\frac{\partial {loss}}{\partial b_2}:= \frac{1}{batch\_size}(1, \cdots ,1) \frac{\partial loss}{\partial u_2}
$$
以及
$$
\frac{\partial {loss}}{\partial x_2}:= \frac{1}{batch\_size}  \frac{\partial loss}{\partial u_2} w_2 ^T \\
\frac{\partial {loss}}{\partial u_1}:=DReLU(u_1)*\frac{\partial {loss}}{\partial x_2}
$$
其中乘法均为安全乘法

Step 6 (反向传播第1层) L, R双方分别将$\frac{\partial {loss}}{\partial u_1}$回复到本地，并分别计算
$$
\frac{\partial {loss}}{\partial w_1 ^L}:= \frac{1}{batch\_size} x_L ^T \frac{\partial loss}{\partial u_1} 
$$
以及
$$
\frac{\partial {loss}}{\partial w_1 ^R}:= \frac{1}{batch\_size} x_R ^T \frac{\partial loss}{\partial u_1}  \\
\frac{\partial {loss}}{\partial b_1}:= \frac{1}{batch\_size}(1, \cdots ,1) \frac{\partial loss}{\partial u_1}
$$

Step 7  L, R根据安全加法协议和明文乘shares协议构造运算图用以更新
$$
w_1 ^L \leftarrow w_1 ^L - learning\_rate * \frac{\partial loss}{\partial w_1 ^L} \\
w_1 ^R \leftarrow  w_1 ^R - learning\_rate * \frac{\partial loss}{\partial w_1 ^R} \\
w_i \leftarrow w_i - learning\_rate * \frac{\partial loss}{\partial w_i}  \mbox{ for }i>=2 \\
b_i \leftarrow  b_i - learning\_rate * \frac{\partial loss}{\partial b_i} 
$$

Step 4. L，R双方运行计算图batch_num次

Step 5.  L方将持有的各参数的share发送给R方，由R方合并出明文。
