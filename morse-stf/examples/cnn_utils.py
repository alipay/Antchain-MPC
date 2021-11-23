"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import pandas as pd
from stensorflow.basic.basic_class.private import PrivateTensor,PrivateTensorBase
#from keras.models import Model


def build_mnist(path):
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape((60000, 28*28), order='C')
    test_images = test_images.reshape((10000, 28 * 28), order='C')
    # 将数据转化为DataFrame格式，方便存储
    training_images = pd.DataFrame(training_images)
    training_labels = pd.DataFrame(training_labels)
    test_images = pd.DataFrame(test_images)
    test_labels = pd.DataFrame(test_labels)
    # 存储到指定路径
    training_images.to_csv(path+"mnist_train_images.csv", index=False, header=None)
    test_images.to_csv(path+"mnist_test_images.csv", index=False, header=None)
    training_labels.to_csv(path+"mnist_train_labels.csv", index=False, header=None)
    test_labels.to_csv(path + "mnist_test_labels.csv", index=False, header=None)


def load_data(normal=False, small=True):
    """
    load mist data
    :param normal: 是否归一化
    :param small: 是否裁剪部分数据
    :return:
    """
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # input channel = 1
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    if normal:
        training_images = training_images / 255.0
        test_images = test_images / 255.0
    else:
        # 转化数据类型
        training_images = training_images.astype(np.int64)
        training_labels = training_labels.astype(np.int64)
        test_images = test_images.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
    if small:
        return training_images[:6400], training_labels[:6400], test_images[:896], test_labels[:896]
    else:
        return training_images, training_labels, test_images, test_labels


def dense_to_onehot(labels_dense, num_classes=10):
    """
    将标签转化为独热码
    :param labels_dense:
    :param num_classes:
    :return:
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    # 展平的索引值对应相加，然后得到精确索引并修改labels_onehot中的每一个值
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot


def convert_datasets(train_x, train_y, test_x, test_y,
                     epoch=1, batch_size=10):
    """
    将原始数据分批加载
    :param train_x: 训练特征
    :param train_y: 训练标签
    :param epoch: 训练轮数
    :param batch_size: 分批加载样本数量
    :return:
    """
    train_y = dense_to_onehot(train_y)
    train_y = train_y.astype(np.int64)
    # 将数据转化s为dataset类型，分批训练
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.repeat(epoch)
    train_dataset = train_dataset.batch(batch_size)
    # 生成迭代器
    iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    image_batch, label_batch = iterator.get_next()
    image_batch = tf.reshape(image_batch, shape=[batch_size, 28, 28, 1])
    label_batch = tf.reshape(label_batch, shape=[batch_size, 10])
    # 测试集也转化为dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
    test_image_batch, test_label_batch = test_iterator.get_next()
    test_image_batch = tf.reshape(test_image_batch, shape=[batch_size, 28, 28, 1])
    test_label_batch = tf.reshape(test_label_batch, shape=[batch_size, 10])
    # 转化数据
    x_train = PrivateTensor(owner='R')
    y_train = PrivateTensor(owner='L')
    x_train.load_from_tf_tensor(image_batch)
    y_train.load_from_tf_tensor(label_batch)
    x_test = PrivateTensor(owner='R')
    y_test = PrivateTensor(owner='L')
    x_test.load_from_tf_tensor(test_image_batch)
    y_test.load_from_tf_tensor(test_label_batch)
    return x_train, y_train, x_test, y_test


def calculate_score(file):
    """
    计算模型准确率
    :param file:
    :return:
    """
    # 加载预测数据
    result = []
    with open(file) as f:
        for line in f:
            line = line.split(',')
            r = [float(x) for x in line]
            # print(r)
            result.append(r.index(max(r)))
    # True label
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    true_label = []
    for i in range(len(result)):
        true_label.append(test_labels[i])
    print("start acc")
    print(len(result))
    print(result)
    print(true_label)
    print(accuracy_score(true_label, result))


def compare_weight(keras_model_path, stf_model_path):
    res = np.load(stf_model_path, allow_pickle=True)
    res = res['weight']
    keras_model = tf.keras.models.load_model(keras_model_path)
    keras_weight = keras_model.get_weights()
    for i in range(len(res)):
        print(res[i].shape)
        print(keras_weight[i][0])
        print(res[i][0])
        print("****")


def LeNet_network():
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)

    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(50, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(10, name="Dense"),
        tf.keras.layers.Activation('softmax')
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=10)
    test_loss = model.evaluate(test_images, test_labels)
    print(test_loss)
    # 全数据，sgd, lr= 0.1, epoch=15, 准确率0.89
    # 全数据，adam,lr= 0.1, epoch=10, 准确率0.91





