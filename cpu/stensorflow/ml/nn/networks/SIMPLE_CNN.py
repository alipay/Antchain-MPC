from stensorflow.ml.nn.networks.NN import NN
from stensorflow.ml.nn.layers.layer import Layer
from stensorflow.ml.nn.layers.input import Input
from stensorflow.ml.nn.layers.relu import ReLU
from stensorflow.ml.nn.layers.conv2d import Conv2d
from stensorflow.ml.nn.layers.pooling import AveragePooling2D
from stensorflow.ml.nn.layers.loss import CrossEntropyLossWithSoftmax
from stensorflow.ml.nn.layers.flatten import Flatten
from stensorflow.ml.nn.layers.dense import Dense, Dense_Local
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.random import random
import numpy as np


class SIMPLE_CNN(NN):
    def __init__(self, feature, label, loss=None):
        super(SIMPLE_CNN, self).__init__()
        # input layer, init data；
        # 这里将dim设置位输入的wight,后续不使用；仅仅是为了应用原有的模板
        layer = Input(dim=28, x=feature)
        local_layer_owner = layer.owner
        self.addLayer(layer)
        # convolutional layer with 1 input channel, 16 output channels and a 5×5 filter
        layer = Conv2d(output_dim=None, fathers=[layer], filters=16,
                       kernel_size=5, input_shape=[28, 28, 1])
        # layer = Conv2d_Local(output_dim=28, fathers=[layer], filters=16,
        #                kernel_size=5, input_shape=[28, 28, 1],owner=local_layer_owner)
        self.addLayer(layer)

        # Relu Layer
        layer = ReLU(output_dim=layer.output_dim, fathers=[layer])
        # layer = ReLU_Local(output_dim=28, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)

        # Average pool
        layer = AveragePooling2D(output_dim=None, fathers=[layer], pool_size=(2, 2))
        # layer = AveragePooling2D_Local(output_dim=None, fathers=[layer], pool_size=(2, 2), owner=local_layer_owner)
        self.addLayer(layer)

        # flatten data, only consider data_format = "NWHC"
        # 这里需要给出正确的output_dim，方便后续的全连接层
        layer = Flatten(output_dim=None, fathers=[layer])
        # layer = Flatten_Local(output_dim=2304, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 临时添加测试
        # layer = Dense(output_dim=100, fathers=[layer])
        # self.addLayer(layer)
        # layer = ReLU(output_dim=100, fathers=[layer])
        # self.addLayer(layer)
        # a 256 × 10 linear layer
        # layer = Dense(output_dim=10, fathers=[layer])
        layer = Dense_Local(output_dim=10, fathers=[layer], owner=local_layer_owner)
        self.addLayer(layer)
        # 输出层
        layer_label = Input(dim=10, x=label)
        self.addLayer(ly=layer_label)
        # 损失计算
        layer_loss = CrossEntropyLossWithSoftmax(layer_score=layer, layer_label=layer_label)
        self.addLayer(ly=layer_loss)

    def predict(self, x, out_prob=True):
        self.cut_off()
        # 输入层
        l_input = self.layers[0]
        assert isinstance(l_input, Input)
        l_input.replace(x)
        self.layers[0] = l_input
        # 输出层
        ly = self.layers[-1]
        if not isinstance(ly, Layer):
            raise Exception("l must be a Layer")
        else:
            ly.forward()

        if out_prob:
            return ly.y
        else:
            return ly.score

    def predict_to_file(self, sess, x, predict_file_name,
                        pred_batch_num, model_file_machine,
                        record_num_ceil_mod_batch_size,
                        with_sigmoid):
        y_pred = self.predict(x=x,  out_prob=with_sigmoid)
        id_y_pred = y_pred.to_tf_str(owner=model_file_machine)
        random.random_init(sess)
        # sess.run(tf.compat.v1.global_variables_initializer())
        # with open(predict_file_name, "w") as f:
        #     for batch in range(batch_num - 1):
        #         records = sess.run(id_y_pred)
        #         records = "\n".join(records.astype('str'))
        #         f.write(records + "\n")
        #
        #     records = sess.run(id_y_pred)[0:record_num_ceil_mod_batch_size]
        #     records = "\n".join(records.astype('str'))
        #     f.write(records + "\n")
        # 分批写入文件
        with open(predict_file_name, "w") as f:
            for batch in range(pred_batch_num):
                records = sess.run(id_y_pred)
                records = "\n".join(records.astype('str'))
                # records.to_file()
                f.write(records + "\n")

    def print(self, model_file_machine, sess):
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Conv2d):
                print(ly)
                for weight in ly.w:
                    print(weight)
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    print("***")
                    print(sess.run(weight))

    def replace_weight(self, keras_weight):
        i = 0
        for ly in self.layers:
            if isinstance(ly, Conv2d):
                kernel = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i].shape)
                kernel.load_from_numpy(keras_weight[i])
                ly.w[0] = kernel
                # 分割
                # kernel = ly.w[0]
                # assert isinstance(kernel, SharedVariablePair)
                # kernel.load_from_tf_tensor(keras_weight[i])
                i += 1
            if isinstance(ly, Dense):
                kernel1 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i].shape)
                kernel1.load_from_numpy(keras_weight[i])
                ly.w[0] = kernel1
                kernel2 = SharedPair(ownerL="L", ownerR="R", shape=keras_weight[i+1].shape)
                kernel2.load_from_numpy(keras_weight[i+1])
                ly.w[1] = kernel2
                # 分割
                # kernel1 = ly.w[0]
                # assert isinstance(kernel1, SharedVariablePair)
                # kernel1.load_from_tf_tensor(keras_weight[i])
                # kernel2 = ly.w[1]
                # assert isinstance(kernel2, SharedVariablePair)
                # kernel2.load_from_tf_tensor(keras_weight[i+1])
                i += 2

    def save_model(self, sess, save_file_path, model_file_machine):
        res = []
        for ly in self.layers:
            if isinstance(ly, Dense) or isinstance(ly, Conv2d):
                for weight in ly.w:
                    weight = weight.to_tf_tensor(owner=model_file_machine)
                    weight = sess.run(weight)
                    res.append(weight)
        res = np.array(res)
        np.savez(save_file_path, weight=res)


    def compare(self, keras_weight, model_file_machine, sess):
        i = 0
        for ly in self.layers:
            if isinstance(ly, Conv2d):
                # 转为tensor
                w = ly.w[0].to_tf_tensor(owner=model_file_machine)
                # 转为numpy
                w = sess.run(w)
                # 比较两个numpy的差距
                res = np.abs(w-keras_weight[i])
                res = res > 1
                if res.any():
                    print("conv2d 发生变化")
                i += 1

            if isinstance(ly, Dense):
                w1 = ly.w[0].to_tf_tensor(owner=model_file_machine)
                w1 = sess.run(w1)
                res1 = np.abs(w1-keras_weight[i])
                res1 = res1 > 5
                if res1.any():
                    print("dense 发生变化")
                i += 2
        pass