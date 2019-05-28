import tensorflow as tf
import numpy as np
import logging
import os
import cifar10_tfrecords
from utils import *
import matplotlib.pyplot as plt
from att_model import Vgg
import skimage
import skimage.transform
import skimage.io


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('seed', 2, "initial random seed")
batch_size = 64
phase = tf.placeholder(tf.bool)

# log info
def set_log_info():
    logger = logging.getLogger('vgg')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler('att.log')
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger


logger = set_log_info()

# set the placeholder
kp_07 = tf.placeholder(tf.float32)
kp_06 = tf.placeholder(tf.float32)
kp_05 = tf.placeholder(tf.float32)

# cifar10 data load
train_image, train_label = cifar10_tfrecords.read_cifar10(data_dir='cifar-10-batches-bin', is_train=True)
train_images, train_labels = cifar10_tfrecords.generate_batch([train_image, train_label], batch_size=batch_size, shuffle=True)
test_image, test_label = cifar10_tfrecords.read_cifar10(data_dir='cifar-10-batches-bin', is_train=False)
test_images, test_labels = cifar10_tfrecords.generate_batch([test_image, test_label], batch_size=batch_size,
                                                                           shuffle=False)

X_input = tf.cond(phase, lambda: train_images, lambda: test_images)
Y_input = tf.cond(phase, lambda: train_labels, lambda: test_labels)

image,p1,p2,p3,loss,lossXent,lossL2,merged,train_op,accuracy = Vgg(X_input,Y_input,phase=phase)


# start session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # start coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))

    for i in range(100000):
        train_feed = {phase: True, kp_07: 0.7, kp_06: 0.6, kp_05: 0.5}
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
            test_feed = {phase: False, kp_07: 1.0, kp_06: 1.0, kp_05: 1.0}
            eval_num = 10000 // batch_size
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

    coord.request_stop()
    coord.join(threads)
