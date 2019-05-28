import tensorflow as tf
from utils import *


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
    assert pool7.get_shape().as_list()[1:] == [1, 1, 512]

    #pool7_flatten = tf.reshape(pool7, [-1, 3 * 3 * 512])



    #global feature
    fc6 = fc_layer(pool7, "fc6", w_shape=[512, 512], b_shape=[512])
    assert fc6.get_shape().as_list()[1:] == [512]
    relu6 = tf.nn.relu(fc6)
    # fc7 = fc_layer(relu6, "fc7", w_shape=[512, 256], b_shape=[256])
    # relu7 = tf.nn.relu(fc7)


    #compute local feature with global feature
    channel1 = conv3_3.shape[3]
    channel2 = conv4_3.shape[3]
    channel3 = conv5_3.shape[3]

    g1 = trans_layer(relu6, 'trans_g1', w_shape=[512, channel1], b_shape=[channel1]) #(batch,256)
    g2 = trans_layer(relu6, 'trans_g2', w_shape=[512, channel2], b_shape=[channel2]) #(batch,512)
    g3 = trans_layer(relu6, 'trans_g3', w_shape=[512, channel3], b_shape=[channel3]) #(batch,512)

    ga_1,p1 = compatibility_func(conv3_3, g1) #(batch,56,56) (batch,56,56)
    ga_2,p2 = compatibility_func(conv4_3, g2)
    ga_3,p3 = compatibility_func(conv5_3, g3)

    ga_total = tf.concat([ga_2,ga_3],axis = 1)

    logit = fc_layer(ga_total, "cls", w_shape=[ga_total.shape[1], 10], b_shape=[10])

    softmax = tf.nn.softmax(logit, name='prob')
    y_oh = tf.one_hot(y, depth=10)
    lossXent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logit))
    #correct_prediction = tf.equal(tf.cast(tf.argmax(softmax, 1), tf.int64), y)
    correct_prediction = tf.nn.in_top_k(logit, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    var_list = tf.trainable_variables()
    #L2 regularization
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name]) * 5e-4
    loss = lossXent + lossL2


    tf.summary.scalar('loss',loss)
    tf.summary.scalar('lossXent',lossXent)
    tf.summary.scalar('lossL2',lossL2)
    tf.summary.scalar('train_acc',accuracy)


    global_step = tf.Variable(0,trainable= False)

    # train_op
    learning_rate = tf.train.exponential_decay(
        1.0,
        global_step,
        10000,
        0.5,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-10)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step)

    merged = tf.summary.merge_all()

    return  rgb,p1,p2,p3,loss,lossXent,lossL2,merged,train_op,accuracy
