"""
      Ant Group
   Copyright (c) 2004-2023 All Rights Reserved.
   ------------------------------------------------------
   File Name : run_vgg16_ti.py
   Author : Qizhi Zhang, Yu Zheng
   Email: qizhi.zqz@antgroup.com
   Create Time : 2021/9/2 上午11:08
   Description : load data
"""

# for Tiny-ImageNet
import os
# from PIL import Image
# from keras.utils import np_utils

from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from stensorflow.basic.basic_class.private import PrivateTensor,PrivateTensorBase
from tensorflow.python.keras.backend import spatial_2d_padding
from stensorflow.global_var import StfConfig

#from keras.models import Model


def build_mnist(path):
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images, training_labels = shuffle(training_images, training_labels)
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


def load_data_mnist(normal=False, small=True):
    """
    load mist data
    :param normal: 是否归一化
    :param small: 是否裁剪部分数据
    :return:
    """
    # mnist = tf.keras.datasets.fashion_mnist
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # input channel = 1
    training_images, training_labels = shuffle(training_images, training_labels)
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    if normal:
        training_images = training_images / 255.0
        test_images = test_images / 255.0
        training_images = (training_images-0.1307)/0.3081
        test_images = (test_images-0.1307)/0.3081

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


def load_data_cifar10(normal=False, small=True):
    """
    load mist data
    :param normal: 是否归一化
    :param small: 是否裁剪部分数据
    :return:
    """
    # mnist = tf.keras.datasets.fashion_mnist
    cifar10 = tf.keras.datasets.cifar10
    (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
    print((training_images.shape, training_labels.shape), (test_images.shape, test_labels.shape))
    # input channel = 1
    training_images, training_labels = shuffle(training_images, training_labels)
    training_images = training_images.reshape(50000, 32, 32, 3)
    test_images = test_images.reshape(10000, 32, 32, 3)
    max_value = np.max(training_images)
    if normal:
        training_images = training_images / max_value
        test_images = test_images / max_value
        mean = np.mean(training_images)
        std = np.std(training_images)
        print("mean=", mean)
        print("std=", std)
        training_images = (training_images-mean)/std
        test_images = (test_images-mean)/std
        training_images = training_images - mean
        test_images = test_images - mean


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


def get_annotations_map():
    # Use your absolute path
    valAnnotationsPath = StfConfig.stf_home + '/dataset/tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations


def load_images(path, num_classes):
    # Load images

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')

    trainPath = path + '/train'

    print('loading training images...');

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        if sChild[0] != 'n':
            continue
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c), mode="r"))
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        print(j)
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images ', str(i))

    return X_train, y_train, X_test, y_test


def load_data_tiny_imagenet(normal=False, small=True):
    """
    load mist data
    :param normal: normalization
    :param small:
    :return:
    """
    num_classes = 200
    # Use your absolute path
    img_path = StfConfig.stf_home + '/dataset/tiny-imagenet-200'
    x_train, y_train, x_test, y_test = load_images(img_path, num_classes)

    training_images = x_train.astype('float32') / 255
    # print("l209 max=", np.max(training_images))
    test_images = x_test.astype('float32') / 255
    # test_labels = np_utils.to_categorical(y_test, num_classes)

    training_images = training_images.reshape(100000, 64, 64, 3)
    training_labels = y_train.reshape(100000, 1)
    test_images = test_images.reshape(10000, 64, 64, 3)
    test_labels = y_test.reshape(10000, 1)

    # # (training_images, training_labels), (test_images, test_labels) = tiny_imagenet.load_data()
    # # training_images, test_images, training_labels, test_labels = X_train, y_train, X_test, y_test
    # print("tiny_imagenet is,")
    # # input channel = 1
    # if small:
    #     return training_images[:6400], training_labels[:6400], test_images[:896], test_labels[:896]
    # else:
    #     return training_images, training_labels, test_images, test_labels
    max_value = np.max(training_images)
    if normal:
        training_images = training_images / max_value
        test_images = test_images / max_value
        mean = np.mean(training_images)
        std = np.std(training_images)
        print("mean=", mean)
        print("std=", std)
        training_images = (training_images - mean) / std
        test_images = (test_images - mean) / std
        training_images = training_images - mean
        test_images = test_images - mean


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


def convert_datasets(train_x, train_y, test_x, test_y, size=(28, 28, 1),
                     epoch=1, batch_size=10, classes_num=10, padding=None):
    """
    将原始数据分批加载
    :param train_x: 训练特征
    :param train_y: 训练标签
    :param epoch: 训练轮数
    :param batch_size: 分批加载样本数量
    :return:
    """
    # 将数据转化s为dataset类型，分批训练
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    def gen_train():
        for x,y in zip(train_x, train_y):
            yield (x,y)
    def gen_test():
        for x,y in zip(test_x, test_y):
            yield (x,y)

    with tf.device(StfConfig.workerR[0]):
        train_dataset = tf.data.Dataset.from_generator(gen_train, output_types=(tf.int64, tf.int64))
        train_dataset = train_dataset.repeat(int(1+epoch))
        train_dataset = train_dataset.batch(batch_size)
        # 生成迭代器
        iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
        image_batch, label_batch = iterator.get_next()
        image_batch = tf.reshape(image_batch, shape=[batch_size, size[0], size[1], size[2]])
        label_batch = tf.one_hot(label_batch, depth=classes_num)
        if padding is not None:
            image_batch = spatial_2d_padding(image_batch, padding=padding)
        label_batch = tf.reshape(label_batch, shape=[batch_size, classes_num])

        # 测试集也转化为dataset
        # test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_generator(gen_test, output_types=(tf.int64, tf.int64)).batch(batch_size)
        test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
        test_image_batch, test_label_batch = test_iterator.get_next()
        test_image_batch = tf.reshape(test_image_batch, shape=[batch_size, size[0], size[1], size[2]])
        if padding is not None:
            test_image_batch = spatial_2d_padding(test_image_batch, padding=padding)
        test_label_batch = tf.reshape(test_label_batch, shape=[batch_size, classes_num])
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


def calculate_score_mnist(file):
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
    # mnist = tf.keras.datasets.fashion_mnist
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    true_label = []
    for i in range(len(result)):
        true_label.append(test_labels[i])
    print("start acc")
    print(len(result))
    # print(result)
    # print(true_label)
    print(accuracy_score(true_label, result))


def calculate_score_cifar10(file):
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
    cifar10 = tf.keras.datasets.cifar10
    (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
    true_label = []
    for i in range(len(result)):
        true_label.append(test_labels[i][0])
    print("start acc")
    print(len(result))
    # print(result)
    # print(true_label)
    print(accuracy_score(true_label, result))




def calculate_score_tiny_imagenet(file):
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
    # mnist = tf.keras.datasets.fashion_mnist
    # mnist = tf.keras.datasets.mnist
    # (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    num_classes = 200
    # Use your absolute path
    img_path = StfConfig.stf_home + '/dataset/tiny-imagenet-200'
    x_train, y_train, x_test, y_test = load_images(img_path, num_classes)
    true_label = []
    for i in range(len(result)):
        true_label.append(y_test[i])
    print("start acc")
    print(len(result))
    # print(result)
    #print(true_label)
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





if __name__ == '__main__':
    load_data_cifar10(normal=True, small=True)