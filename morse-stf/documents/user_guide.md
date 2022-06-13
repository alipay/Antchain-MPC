# User Guide
[toc]

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

#### PrivateTensor

`PrivateTensor` represnts private data owned by one party. It contains an `inner_value` of type `tf.Tensor of dtype = int64`, `int module` , `int fixedpoint`, and `owner`.  
When `module is None`, a fixed-point number is defined, which represents `inner_value*pow(2,-fixedpoint)`. When `module is not None`, it defines a tensor of `module`-factorial cyclic group.

#### SharedTensor 
`SharedTensor`contains `inner_value` and `module`.

#### SharedPair 
`SharedPair` contains `xL`, `xR`, `ownerL`, `ownerR`, `fixedpoint`. Both `(xL, xR)` are `SharedTensor`. In particular, `ownerL != ownerR` and `xL.module == xR.module`.  
When `module is None`, a fixed-point number is defined, which represents `(xL.inner_value + xR.inner_value mod pow(2,64)) * pow(2, -fixedpoint)`. When `module is not None`, it defines a tensor of `module`-factorial cyclic group, which represents `(xL.inner_value + xR.inner_value mod module)`.

### Load Data
`PrivateTensor` can load data from a (local or federated) server.
To get 'const PrivateTensor', you can use `load_from_numpy()`, `load_from_tf_tensor`.
To get `non-const PrivateTensor`, you can use `load_first_line_from_file()`, `load_from_file`, `load_from_file_withid()`

You can load data using `load_from_file()` by following the example below.

```
    x_train=PrivateTensor(owner='L')
    format_x=[["a"],[0.1],[0.1],[0.1],[0.1],[0.1]]
    x_train.load_from_file(path=path, record_defaults=format_x, batch_size=batch_size, repeat=repeat, skip_col_num=1)
```

`path` is absolute path; `format_x` is data type; `batch_size` is the size of the loading data at each batch; `repeat` is the times of reading data; `skip_col_num` is the number of collums that are skiped.

### Build the model

#### Linear Regression

You can build linear regression with one-party private inputs by:
```
    fromstensorflow.ml.logistic_regressionimportLogisticRegression
    model=LogisticRegression(num_features=featureNum, learning_rate=learning_rate)
```
`num_features` is the number of features; `learning_rate` is the learning rate of model training.

You can build linear regression with two-party private inputs by:
```
    fromstensorflow.ml.logistic_regression2importLogisticRegression2
    model=LogisticRegression2(learning_rate=learning_rate, num_features_L=featureNumL, num_features_R=featureNumR)
```
`learning_rate` is the learning rate of model training; `num_features_L` is the number of private features owned by party `workerL`; `num_features_R` is the number of private features owned by party `workerR`.

#### Fully Connected Model 

You can build the fully connected model with one-party private inputs by:
```
    fromstensorflow.ml.nn.networks.DNNimportDNN
    model=DNN(feature=x_train,label=y_train,dense_dims=dense_dims)
    model.compile()
```
`feature` is a `PrivateTensor` representing features; `label` is a `PrivateTensor` representing labels; `dense_dims` is a `list` of type `int`, representing the number of neurons at each layer.

You can build fully connected model (FCM) with two-party private inputs by:
```
    fromstensorflow.ml.nn.networks.DNNimportDNN
    model=DNN(feature=xL_train, label=y_train, dense_dims=dense_dims, feature_another=xR_train)
    model.compile()
```
Both `feature`, `feature_another` are `PrivateTensor` representing features; `label` is a `PrivateTensor` representing labels; `dense_dims` is a `list` of type `int`, representing the number of neurons at each layer.

#### Convolutional Neural Network 
You can build Convolutional Neural Network (CNN) with one-party private inputs by:
```
    fromstensorflow.ml.nn.networks.NETWORKBimportNETWORKB
    model=NETWORKB(feature=x_train, label=y_train, loss="CrossEntropyLossWithSoftmax")
    model.compile()
```
`feature` is a `PrivateTensor` representing features; `label` is a `PrivateTensor` representing labels; `loss` defines the loss function.
The layers for `NETWORKB` are described below,
```
    Conv2D(16,(5,5),activation='relu',use_bias=False),
    AvgPool2D(2,2),
    Conv2D(16,(5,5),activation='relu',use_bias=False),
    AvgPool2D(2,2),
    Flatten(),
    Dense(100,activation='relu'),
    Dense(10,name="Dense"),
    Activation('softmax')
```
`NETWORKA`, `NETWORKC`, and `NETWORKD` follow the same format. See [here]().

### Model Training

#### Initialization

Before training a model, you need to initialize `Session` and `Variable`.
```
     sess=tf.compat.v1.Session(StfConfig.target)
     init_op=tf.compat.v1.initialize_all_variables()
     sess.run(init_op)
```

#### Start Training 

You can train linear regression with one-party private inputs by
```
     model.fit(sess=sess,x=x_train,y=y_train,num_batches=train_batch_num)
```

You can train linear regression with two-party private inputs by
```
model.fit(sess, x_L=xL_train, x_R=xR_train, y=y_train, num_batches=train_batch_num)
```

You can train FCM/CNN with one/two-party private inputs by
```
    model.train_sgd(learning_rate=learning_rate,batch_num=train_batch_num, l2_regularization=l2_regularization, sess=sess)
```

### Model Prediction

For linear regression with one-party private inputs, you could use
```
    model.predict(id,x_test,pred_batch_num,sess,predict_file=None)
```

For linear regression with two-party private inputs, you could use
```
    model.predict(id, xL_test, xR_test, pred_batch_num, sess, predict_file=None)
```

For fully connected model with one-party private inputs, you could use
```
    model.predict_to_file(sess=sess, x=x_test, predict_file_name=StfConfig.predict_to_file, batch_num=pred_batch_num, idx=id)
```

For fully connected model with two-party private inputs, you could use
```
    model.predict_to_file(sess=sess, x=xL_test, x_another=xR_test, predict_file_name=StfConfig.predict_to_file, batch_num=pred_batch_num, idx=id)
```

For FCN with one-party private inputs, you could use
```
    model.predict_to_file(sess=sess, x=x_test, predict_file_name=StfConfig.predict_to_file, batch_num=pred_batch_num, idx=id)
```

For FCN with two-party private inputs, you could use
```
    model.predict_to_file(sess=sess, x=xL_test, x_another=xR_test, predict_file_name=StfConfig.predict_to_file, batch_num=pred_batch_num, idx=id)
```

For CNN with one-party private inputs, you could use
```
    model.predict_to_file(sess, x_test, predict_file_name=StfConfig.predict_to_file, pred_batch_num=pred_batch_num, with_sigmoid=False)
```

### Choose Your Protocol

Morse-STF allows users to choose their MPC protocols, including `const`, `log`, and `linear`. You could configure `protocols` in `config.json`. For example,
```
    "protocols":{
    	"drelu": "log"
    }
```

Here, we summarize training time (for 1 epoch) of `const`, `log`, and `linear` for your reference.

|  Dataset  | Network    |    Delay (ms) | `linear` (s) | `log` (s) | `const` (s)|
|  ---      | ---        |    ---        |    ---       |    ---    |     ---    |
| xindai10  | 32, 32     |    5          |     67       |    46     |     208    |
| xindai10  | 32, 32     |    10         |     113      |    52     |     208    |
| xindai10  | 32, 32     |    30         |     306      |    94     |     224    |
| xindai10  | 32, 32     |    60         |     606      |    165    |     252    |
| xindai291 |  7, 7      |    5          |     226      |     93    |     204    |
| xindai291 |  7, 7      |    10         |     445      |    133    |     205    |
| xindai291 |  7, 7      |    30         |    1307      |    308    |     253    |
| xindai291 |  7, 7      |    60         |    2549      |    612    |     492    |
| xindai291 | 32, 32     |    5          |     234      |    123    |     928    |
| xindai291 | 32, 32     |    10         |     450      |    134    |     928    |
| xindai291 | 32, 32     |    30         |    1332      |    296    |     979    |
| xindai291 | 32, 32     |    60         |    2657      |    640    |    1284    |
| MNIST     | networka   |    60         |    7996      |    790    |     2527   |
| MNIST     | networkb   |    60         |   19125      |   26109   |     n.a.   |
| MNIST     | networkc   |    60         |   23766      |   33764   |     n.a.   |
| MNIST     | networkd   |    60         |    4620      |    3230   |    25913   |
