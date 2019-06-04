import tensorflow as tf
import numpy as np
import logging
import os
import cifar100_test
from utils import *
import matplotlib.pyplot as plt
from att_model import Vgg
import skimage
import skimage.transform
import skimage.io


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "batch size for training the model")
flags.DEFINE_integer("total_step", 200000, "total step to train the model")
flags.DEFINE_integer('seed', 2, "initial random seed")
flags.DEFINE_string('result_log','att.log','print exp results to log file')

FLAGS = flags.FLAGS

phase = tf.placeholder(tf.bool)

# log info
def set_log_info():
    logger = logging.getLogger('vgg')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler(FLAGS.result_log)
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger


logger = set_log_info()

# set the placeholder
images = tf.placeholder(tf.float32,shape=[None,32,32,3])
labels = tf.placeholder(tf.float32,[None])
kp_07 = tf.placeholder(tf.float32)
kp_06 = tf.placeholder(tf.float32)
kp_05 = tf.placeholder(tf.float32)

# cifar100 data load
cifar100 = Cifar100DataReader(r'C:\Users\caodada\Desktop\cifar100_att\cifar-100-python')

image,p1,p2,p3,loss,lossXent,lossL2,merged,train_op,accuracy = Vgg(images,labels,phase,kp_07,kp_06,kp_05)


# start session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    train_images, train_labels = cifar100.next_train_data

    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))

    for i in range(FLAGS.total_step):
        train_feed = {phase: True, kp_07: 0.7, kp_06: 0.6, kp_05: 0.5,images:train_images,labels:train_labels}
        result = sess.run([image,p1,p2,p3,loss,lossXent,lossL2, train_op, accuracy], feed_dict=train_feed)

        if i and i % 100 == 0:

            logger.info(
                'step {}: loss = {:3.4f}\tlossXent = {:3.4f}\tlossL2 = {:3.4f}'.format(
                    i, result[4],result[5],result[6]))

            logger.info('step {}: train_acc = {:3.4f}'.format(
                i, result[-1], ))


            # attention map visualization
            if i % 500 == 0:
                plt.figure(figsize=[5, 5])
                plt.subplot(221)
                plt.imshow(result[0][0])
                plt.subplot(222)
                plt.imshow(result[0][0])
                img = skimage.transform.pyramid_expand(result[2][0, 0, :].reshape(16, 16), upscale=2, sigma=10)
                plt.imshow(img, alpha=0.8)
                plt.subplot(223)
                plt.imshow(result[0][0])
                img = skimage.transform.pyramid_expand(result[3][0, 0, :].reshape(8, 8), upscale=4, sigma=10)
                plt.imshow(img, alpha=0.8)

                if not os.path.exists('train_imgs'):
                    os.mkdir('train_imgs')
                plt.savefig('train_imgs/train_step%d.png'%i)

        if i and i % 10 == 0:
            summary = sess.run(merged, feed_dict=train_feed)
            writer.add_summary(summary, i)

        # evalresult

        if i and i % 1000 == 0:
            test_images, test_labels = cifar100.next_test_data
            test_feed = {phase: False, kp_07: 1.0, kp_06: 1.0, kp_05: 1.0,images:test_images,labels:test_labels}
            eval_num = 10000 // FLAGS.batch_size
            total_accuracy = 0
            for k in range(eval_num):
                result = sess.run([image,p1,p2,p3,accuracy], feed_dict=test_feed)
                total_accuracy += result[-1]

            acc = total_accuracy / eval_num
            # print('eval_acc',accuracy)
            # print(result[-1])
            logger.info(
                'step {}: Test_acc = {:3.4f}\t acc_batch = {}'.format(
                    i, acc, result[-1]))

            plt.figure(figsize=[5, 5])
            plt.subplot(221)
            plt.imshow(result[0][0])
            plt.subplot(222)
            plt.imshow(result[0][0])
            img = skimage.transform.pyramid_expand(result[2][0, 0, :].reshape(16, 16), upscale=2, sigma=10)
            plt.imshow(img, alpha=0.8)
            plt.subplot(223)
            plt.imshow(result[0][0])
            img = skimage.transform.pyramid_expand(result[3][0, 0, :].reshape(8, 8), upscale=4, sigma=10)
            plt.imshow(img, alpha=0.8)

            if not os.path.exists('test_imgs'):
                os.mkdir('test_imgs')
            plt.savefig('test_imgs/test_step%d.png' % i)
