import pickle   # 用于序列化和反序列化
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math

BATCH_SIZE = 64


class Cifar100DataReader():
    def __init__(self, cifar_folder, onehot=True):
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.data_label_train = None  # 训练集
        self.data_label_test = None  # 测试集
        self.batch_index = 0  # 训练数据的batch块索引
        self.test_batch_index = 0  # 测试数据的batch_size
        f = os.path.join(self.cifar_folder, "train")  # 训练集有50000张图片，100个类，每个类500张
        #print('read: %s' % f)
        fo = open(f, 'rb')
        self.dic_train = pickle.load(fo, encoding='bytes')
        fo.close()
        self.data_label_train = list(zip(self.dic_train[b'data'], self.dic_train[b'fine_labels']))  # label 0~99
        np.random.shuffle(self.data_label_train)

    # def dataInfo(self):
    #     print(self.data_label_train[0:2])  # 每个元素为二元组，第一个是numpy数组大小为32*32*3，第二是label
    #     print(self.dic_train.keys())
    #     print(b"coarse_labels:", len(self.dic_train[b"coarse_labels"]))
    #     print(b"filenames:", len(self.dic_train[b"filenames"]))
    #     print(b"batch_label:", len(self.dic_train[b"batch_label"]))
    #     print(b"fine_labels:", len(self.dic_train[b"fine_labels"]))
    #     print(b"data_shape:", np.shape((self.dic_train[b"data"])))
    #     print(b"data0:", type(self.dic_train[b"data"][0]))

    # 得到下一个batch训练集，块大小为100
    def next_train_data(self, batch_size=BATCH_SIZE):
        """
        return list of numpy arrays [na,...,na] with specific batch_size
                na: N dimensional numpy array
        """
        if self.batch_index < len(self.data_label_train) / batch_size-1:
            #print("batch_index:", self.batch_index)
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.batch_index = 0
            np.random.shuffle(self.data_label_train)
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)


            # 把一个batch的训练数据转换为可以放入神经网络训练的数据

    def _decode(self, datum, onehot):
        rdata = list()  # batch训练数据
        rlabel = list()
        if onehot:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))  # 转变形状为：32*32*3
                hot = np.zeros(100)
                hot[int(l)] = 1  # label设为100维的one-hot向量
                rlabel.append(hot)
        else:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                rlabel.append(int(l))
        return rdata, rlabel


        # 得到下一个测试数据 ，供神经网络计算模型误差用

    def next_test_data(self, batch_size=BATCH_SIZE):
        '''''
        return list of numpy arrays [na,...,na] with specific batch_size
                na: N dimensional numpy array
        '''
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test")
            #print('read: %s' % f)
            fo = open(f, 'rb')
            dic_test = pickle.load(fo, encoding='bytes')
            fo.close()
            data = dic_test[b'data']
            labels = dic_test[b'fine_labels']  # 0 ~ 99
            self.data_label_test = list(zip(data, labels))
            self.batch_index = 0

        if self.test_batch_index < len(self.data_label_test) / batch_size-1:
            #print("test_batch_index:", self.test_batch_index)
            datum = self.data_label_test[self.test_batch_index * batch_size:(self.test_batch_index + 1) * batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.test_batch_index = 0
            np.random.shuffle(self.data_label_test)
            datum = self.data_label_test[self.test_batch_index * batch_size:(self.test_batch_index + 1) * batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)

            # 显示 9张图像

    # def showImage(self):
    #     rdata, rlabel = self.next_train_data()
    #     fig = plt.figure()
    #     ax = fig.add_subplot(331)
    #     ax.imshow(rdata[0])
    #     ax = fig.add_subplot(332)
    #     ax.imshow(rdata[1])
    #     ax = fig.add_subplot(333)
    #     ax.imshow(rdata[2])
    #     ax = fig.add_subplot(334)
    #     ax.imshow(rdata[3])
    #     ax = fig.add_subplot(335)
    #     ax.imshow(rdata[4])
    #     ax = fig.add_subplot(336)
    #     ax.imshow(rdata[5])
    #     ax = fig.add_subplot(337)
    #     ax.imshow(rdata[6])
    #     ax = fig.add_subplot(338)
    #     ax.imshow(rdata[7])
    #     ax = fig.add_subplot(339)
    #     ax.imshow(rdata[8])
    #     plt.show()




#test
# cifar = Cifar100DataReader(r'C:\Users\caodada\Desktop\cifar100_att\cifar-100-python')
# cifar.showImage()
