import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cifar100_test
a = tf.constant([[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],
                 [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]])

b = tf.constant([[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],
                 [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]])

e = tf.constant([[1,4],[2,5],[3,6]])
s = tf.constant([2,3,4])

ss = tf.constant([2,3])

f = tf.constant([[4,5,6],[7,8,9]])
ee = tf.transpose(tf.expand_dims(e[0,:],axis=0),(1,0))

v = [1.0,2.0,3.0]
op = tf.nn.softmax(v)

# c = tf.multiply(e,f)
sess = tf.Session()
# a_eval = sess.run(a)
# c_eval = sess.run(c)
# print(a_eval.shape)
#print(sess.run(tf.reduce_mean(e,axis=1)))
# print(c_eval.shape)


cifar100 = cifar100_test.Cifar100DataReader(r'C:\Users\caodada\Desktop\project\cifar100_att\cifar-100-python')
for i in range(800):
    #cifar100 = cifar100_test.Cifar100DataReader(r'C:\Users\caodada\Desktop\project\cifar100_att\cifar-100-python')
    img,label = cifar100.next_test_data(batch_size=64)

    # print(i,label)
    # plt.imshow(img[0])
    # plt.show()

