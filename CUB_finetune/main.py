import tensorflow as tf
import numpy as np
import logging
import os
from pretrained_utils import *
import matplotlib.pyplot as plt
from tensorflow.core.protobuf import saver_pb2


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "batch size for training the model")
flags.DEFINE_float("learning_rate_start", 1.6, "Learning rate for training the model")
flags.DEFINE_integer("learning_rate_per_epoch", 6000, "learning rate decay every epoch")
flags.DEFINE_float("learning_rate_decay", 0.5, "learning rate decay ratio every epoch")
flags.DEFINE_integer("total_step", 100000, "total step to train the model")
flags.DEFINE_string("checkpoint_dir", 'models/','path to save model parameters')
flags.DEFINE_integer('save_model_per_step',6000,'save model parameters every epoch')
flags.DEFINE_integer('num_channels',3,'the original image number channels')
flags.DEFINE_integer('num_classes',200,'the CUB categories')
flags.DEFINE_string('result_log','cub_att.log','print exp results to log file')
flags.DEFINE_integer('seed', 2, "initial random seed")


# log info
def set_log_info():
    logger = logging.getLogger('Vgg')
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

# inputs
def input(dataname, batchsize, isShuffel, flag):
    image, label = read_and_decode(dataname, flag=flag)
    images, labels = generate_batch([image, label], batchsize, isShuffel)
    return images, labels


phase = tf.placeholder(tf.bool)
kp_07 = tf.placeholder(tf.float32)
kp_06 = tf.placeholder(tf.float32)
kp_05 = tf.placeholder(tf.float32)

def Vgg(rgb,y,phase):
    conv1_1 = ConvBNReLU(rgb, w_shape=[3, 3, 3, 64], b_shape=[64, ], axis=-1, phase=phase, name='conv1_1')
    conv1_1 = tf.nn.dropout(conv1_1, keep_prob=kp_07)
    conv1_2 = ConvBNReLU(conv1_1, w_shape=[3, 3, 64, 64], b_shape=[64, ], axis=-1, phase=phase, name='conv1_2')
    # pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = ConvBNReLU(conv1_2, w_shape=[3, 3, 64, 128], b_shape=[128, ], axis=-1, phase=phase, name='conv2_1')
    conv2_1 = tf.nn.dropout(conv2_1, keep_prob=kp_06)
    conv2_2 = ConvBNReLU(conv2_1, w_shape=[3, 3, 128, 128], b_shape=[128, ], axis=-1, phase=phase, name='conv2_2')
    # pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = ConvBNReLU(conv2_2, w_shape=[3, 3, 128, 256], b_shape=[256, ], axis=-1, phase=phase, name='conv3_1')
    conv3_1 = tf.nn.dropout(conv3_1, keep_prob=kp_06)
    conv3_2 = ConvBNReLU(conv3_1, w_shape=[3, 3, 256, 256], b_shape=[256, ], axis=-1, phase=phase, name='conv3_2')
    conv3_2 = tf.nn.dropout(conv3_2, keep_prob=kp_06)
    conv3_3 = ConvBNReLU(conv3_2, w_shape=[3, 3, 256, 256], b_shape=[256, ], axis=-1, phase=phase, name='conv3_3')
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = ConvBNReLU(pool3, w_shape=[3, 3, 256, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv4_1')
    conv4_1 = tf.nn.dropout(conv4_1, keep_prob=kp_06)
    conv4_2 = ConvBNReLU(conv4_1, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv4_2')
    conv4_2 = tf.nn.dropout(conv4_2, keep_prob=kp_06)
    conv4_3 = ConvBNReLU(conv4_2, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv4_3')
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = ConvBNReLU(pool4, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv5_1')
    conv5_1 = tf.nn.dropout(conv5_1, keep_prob=kp_06)
    conv5_2 = ConvBNReLU(conv5_1, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv5_2')
    conv5_2 = tf.nn.dropout(conv5_2, keep_prob=kp_06)
    conv5_3 = ConvBNReLU(conv5_2, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv5_3')
    pool5 = max_pool(conv5_3, 'pool5')

    conv6_1 = ConvBNReLU(pool5, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv6_1')
    pool6 = max_pool(conv6_1, 'pool6')

    conv7_1 = ConvBNReLU(pool6, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=phase, name='conv7_1')
    pool7 = max_pool(conv7_1, 'pool7')
    assert pool7.get_shape().as_list()[1:] == [3,3,512]

    pool7_flatten = tf.reshape(pool7,[-1,3*3*512])
    #global feature
    fc6 = fc_layer(pool7_flatten, "fc6", w_shape=[pool7_flatten.shape[1], 1024], b_shape=[1024])
    assert fc6.get_shape().as_list()[1:] == [1024]
    fc6 = batch_normalization_layer(fc6, axis=-1, phase=phase, name='fc6/bn')
    relu6 = tf.nn.relu(fc6)
    relu6 = tf.nn.dropout(relu6, keep_prob=kp_05)
    # fc7 = fc_layer(relu6, "fc7", w_shape=[512, 256], b_shape=[256])
    # relu7 = tf.nn.relu(fc7)


    #compute local feature with global feature
    channel1 = conv3_3.shape[3]
    channel2 = conv4_3.shape[3]
    channel3 = conv5_3.shape[3]

    g1 = trans_layer(relu6, 'trans_g1', w_shape=[fc6.get_shape().as_list()[1], channel1], b_shape=[channel1]) #(batch,256)
    g2 = trans_layer(relu6, 'trans_g2', w_shape=[fc6.get_shape().as_list()[1], channel2], b_shape=[channel2]) #(batch,512)
    g3 = trans_layer(relu6, 'trans_g3', w_shape=[fc6.get_shape().as_list()[1], channel3], b_shape=[channel3]) #(batch,512)

    # ga_1, p1 = compatibility_func(conv3_3, g1)  # (batch,56,56) (batch,56,56)
    # ga_2, p2 = compatibility_func(conv4_3, g2)
    # ga_3, p3 = compatibility_func(conv5_3, g3)

    ga_1,p1 = batch_process(conv3_3, g1) #(batch,56,56) (batch,56,56)
    ga_2,p2 = batch_process(conv4_3, g2)
    ga_3,p3 = batch_process(conv5_3, g3)

    ga_total = tf.concat([ga_1,ga_2,ga_3],axis = 1)

    logit = fc_layer(ga_total, "cls", w_shape=[ga_total.shape[1], FLAGS.num_classes], b_shape=[FLAGS.num_classes])

    softmax = tf.nn.softmax(logit, name='prob')
    y_oh = tf.one_hot(y, depth=FLAGS.num_classes)
    lossXent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logit))
    correct_prediction = tf.equal(tf.cast(tf.argmax(softmax, 1), tf.int64),y)
    #correct_prediction = tf.nn.in_top_k(logit, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # L2 regularization
    var_list = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name]) * 5e-4
    loss = lossXent + lossL2


    global_step = tf.Variable(0,trainable= False)

    # train_op
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate_start,
        global_step,
        FLAGS.learning_rate_per_epoch,
        FLAGS.learning_rate_decay,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-4)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss,global_step=global_step)
        train_op2 = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss,global_step=global_step)
        train_op3 = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(loss,global_step=global_step)
        train_op4 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)


    tf.summary.scalar('loss', loss)
    tf.summary.scalar('lossXent', lossXent)
    tf.summary.scalar('lossL2', lossL2)
    tf.summary.scalar('train_acc', accuracy)
    tf.summary.scalar('learning rate',learning_rate)

    merged = tf.summary.merge_all()

    return  rgb,p1,p2,p3,loss,lossXent,lossL2,merged,train_op1,train_op2,train_op3,train_op4,global_step,accuracy




#save and restore
def save_checkpoint(sess,step,saver):
    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess=sess,
               save_path=checkpoint_dir+'model.ckpt',
               global_step=step)
    print('step:%d | save model success'%step)

def load_checkpoint(sess,saver):
    checkpoint_dir = FLAGS.checkpoint_dir
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints and checkpoints.model_checkpoint_path:
        #checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
        #saver.restore(sess, os.path.join(checkpoint_dir,checkpoints_name))
        saver.restore(sess,checkpoints.model_checkpoint_path)
        #step = str(10001)
        #saver.restore(sess,checkpoint_dir+"model.ckpt-"+step)
        print('load model success,contuinue training...')
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('None checkpoint file found,initialize model... ')



train_images, train_labels = input('train.tfrecords', batchsize=FLAGS.batch_size, isShuffel=True,
                                   flag=True)
test_images, test_labels = input('test.tfrecords', batchsize=FLAGS.batch_size, isShuffel=True,
                                 flag=False)


X_input = tf.cond(phase, lambda: train_images, lambda: test_images)
Y_input = tf.cond(phase, lambda: train_labels, lambda: test_labels)

image,p1,p2,p3,loss,lossXent,lossL2,merged,train_op1,train_op2,train_op3,train_op4,global_step,accuracy = Vgg(X_input,Y_input,phase=phase)



with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)

    saver = tf.train.Saver(max_to_keep=15,write_version=saver_pb2.SaverDef.V1)
    load_checkpoint(sess, saver)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))

    for i in range(FLAGS.total_step):

        train_feed = {phase: True, kp_07: 0.7, kp_06: 0.6, kp_05: 0.5}


        if i and i % 10 == 0:
            summary,step = sess.run([merged,global_step],feed_dict=train_feed)
            writer.add_summary(summary, step)

        if i < 6000:
            result = sess.run([image,p1,p2,p3,loss,lossXent,lossL2, train_op1,global_step,accuracy],
                              feed_dict=train_feed)
        elif i < 12000:
            result = sess.run([image, p1, p2, p3, loss, lossXent, lossL2, train_op2, global_step, accuracy],
                              feed_dict=train_feed)
        elif i < 18000:
            result = sess.run([image, p1, p2, p3, loss, lossXent, lossL2, train_op3, global_step, accuracy],
                              feed_dict=train_feed)
        else:
            result = sess.run([image, p1, p2, p3, loss, lossXent, lossL2, train_op4, global_step, accuracy],
                              feed_dict=train_feed)

        if i and i % 100 == 0:

            logger.info(
                'step {}: loss = {:3.4f}\tlossXent = {:3.4f}\tlossL2 = {:3.4f}'.format(
                    result[-2], result[4],result[5],result[6]))

            logger.info('step {}: train_acc = {:3.4f}'.format(
                result[-2], result[-1], ))

            if i % 500 == 0:
                plt.figure(figsize=[5,5])
                plt.subplot(221)
                plt.imshow(result[0][0])
                plt.subplot(222)
                plt.imshow(result[1][0])
                plt.subplot(223)
                plt.imshow(result[2][0])
                plt.subplot(224)
                plt.imshow(result[3][0])
                if not os.path.exists('train_imgs_1212'):
                    os.mkdir('train_imgs_1212')
                plt.savefig('train_imgs_1212/train_step%d.png'%i)

        if i and i % FLAGS.save_model_per_step == 0:
            save_checkpoint(sess,result[-2],saver)


        # eval result

        if i and i % 1000 == 0:
            eval_num = 10000 // FLAGS.batch_size
            total_accuracy = 0
            test_feed = {phase: False, kp_07: 1.0, kp_06: 1.0, kp_05: 1.0}

            for i in range(eval_num):
                result = sess.run([image,p1,p2,p3,global_step,accuracy], feed_dict=test_feed)
                total_accuracy += result[-1]


            acc = total_accuracy / eval_num
            # print('eval_acc',accuracy)
            # print(result[-1])
            logger.info(
                'step {}: Test_acc = {:3.4f}\t acc_batch = {}'.format(
                    result[-2], acc, result[-1]))

            plt.figure(figsize=[5,5])
            plt.subplot(221)
            plt.imshow(result[0][0])
            plt.subplot(222)
            plt.imshow(result[1][0])
            plt.subplot(223)
            plt.imshow(result[2][0])
            plt.subplot(224)
            plt.imshow(result[3][0])
            if not os.path.exists('test_imgs'):
                os.mkdir('test_imgs')
            plt.savefig('test_imgs/test_step%d.png' % i)


    save_checkpoint(sess, total_step, saver)


    coord.request_stop()
    coord.join(threads)
