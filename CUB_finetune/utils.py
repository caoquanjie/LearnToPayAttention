from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


paras = np.load('attention_vgg.npy', encoding='latin1').item()


def _parameter_summary(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.reduce_sum(tf.pow(tf.norm(params, axis=(0,1)), 2), axis=1))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _parameter_summary_fc(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.pow(tf.norm(params, axis=1), 2))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _output_summary(outputs):
    tf.summary.histogram(outputs.op.name + '/outputs', outputs)
    tf.summary.scalar(outputs.op.name + '/outputs_sparsity',
		tf.nn.zero_fraction(outputs))


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name=name)


def conv_layer_BN(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, w_shape)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        _parameter_summary_fc(filt)
        _output_summary(bias)

    return bias

def relu_layer(x,name):
    with tf.variable_scope(name):
        h = tf.nn.relu(x,name=name)
        _output_summary(h)

    return h



def batch_normalization_layer(x,axis,phase,name):
    with tf.variable_scope(name):
        h = tf.layers.batch_normalization(x,axis=axis,training=phase,name=name)
    return h



def ConvBNReLU(bottom,w_shape,b_shape,axis,phase,name):
    with tf.variable_scope(name):
        h = conv_layer_BN(bottom,name=name+'/conv',w_shape=w_shape,b_shape=b_shape)

    with tf.variable_scope(name):
        h = batch_normalization_layer(h,axis=axis,phase=phase,name=name+'/BN')

    with tf.variable_scope(name):
        h = relu_layer(h,name = name+'/ReLU')

    return h


def conv_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, w_shape)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)

        _parameter_summary_fc(filt)
        _output_summary(relu)
        return relu



def fc_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name, w_shape)
        biases = get_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)

        return fc


def get_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0))


def get_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def get_conv_filter(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def get_bias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0.1))


def get_trans_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0.1))


def get_trans_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

def trans_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_trans_fc_weight(name, w_shape)
        biases = get_trans_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)
        return fc

def read_and_decode(filename, flag):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           # 'channel':tf.FixedLenFeature([],tf.int64)
                                       })  # 将image数据和label取出来
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    # channel = tf.cast(features['channel'],tf.int32)
    if flag:

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, tf.stack([height, width, 3]))  # reshape为448*448的3通道图片

        img = tf.random_crop(img, [80, 80, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32) * (1. / 255)  # 在流中抛出img张量
    else:
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, tf.stack([height, width, 3]))  # reshape为448*448的3通道图片
        img = tf.image.resize_image_with_crop_or_pad(img, 80, 80)
        img = tf.cast(img, tf.float32) * (1. / 255)

    label = tf.cast(features['label'], tf.int64)  # 在流中抛出label张量
    return img, label

# 生成一个batch的数据集，返回
def generate_batch(example, batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=20 * batch_size,
            min_after_dequeue=10 * batch_size)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            capacity=10 * batch_size)
    return ret

# def compatibility_func(L,g):
#     size = L.shape[1]
#     batch = L.shape[0]
#     #L = tf.reshape(L,[batch,size * size,channel])
#     for i in range(batch):
#         score = tf.multiply(L[i,:,:,:],g[i,:]) #(56,56,256)
#         map = tf.reduce_sum(score,axis=2) #(56,56)
#         score_map = tf.reshape(map,[-1,1])
#         score_vector = tf.nn.softmax(score_map)
#         img = tf.reshape(score_vector,[size,size])
#         tf.add_to_collection('scoreVector%d'%size,img)  #(batch,56,56)
#         tf.add_to_collection('img%d'%size,img)
#     scoreVector = tf.get_collection('scoreVector%d'%size)   #(batch,56,56)
#     image = tf.get_collection('img%d'%size) #(batch,56,56)
#     scoreVector = tf.convert_to_tensor(scoreVector)
#     image = tf.convert_to_tensor(image)
#     print(scoreVector.shape)
#
#     return scoreVector,image

def compatibility_func(L,g):
    size = L.shape[1]
    batch = L.shape[0]
    for i in range(batch):
        L_vector = tf.reshape(L[i,:,:,:],[size*size,-1]) #(56*56,256)
        g_vector = tf.expand_dims(g[i,:],axis=0)
        g_vectorT = tf.transpose(g_vector,(1,0))
        score_vector = tf.matmul(L_vector,g_vectorT) #(56*56,1)
        score = tf.nn.softmax(score_vector)
        img = tf.reshape(score,[size,size])
        a_vector = tf.squeeze(score)
        L_vectorT = tf.transpose(L_vector, (1, 0))  # (256,56*56)
        gas = tf.multiply(L_vectorT, a_vector)  # (256,56*56)
        tf.add_to_collection('gas%d' % size, gas)
        #tf.add_to_collection('img%d'%size,img)
    #image = tf.get_collection('img%d'%size) #(batch,56,56)
    ga = tf.get_collection('gas%d' % size)  # (batch,256,56*56)
    ga = tf.convert_to_tensor(ga)
    ga = tf.reduce_mean(ga,[2])
    print(ga.shape)
    return ga,img


#test
with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  #sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads= tf.train.start_queue_runners(coord=coord)
  image, label = read_and_decode('test.tfrecords',flag=False)
  #image = preprocess_for_train(image,image_size = 160)

  '''images, labels = generate_batch([image, label], 30, shuffle=True)
  images = tf.cast(images, dtype=tf.float32)
  labels = tf.cast(labels, dtype=tf.int64)'''
  threads = tf.train.start_queue_runners(coord = coord)


  for i in (range(50)):


      #images, labels = generate_batch([image, label], 10000, 32, shuffle=True)

      example,l = sess.run([image,label])
      print(i,l,example)

      plt.imshow(example)
      plt.show()


      #img=Image.fromarray(example, 'RGB')#这里Imag e是之前提到的
        #example.save('/home/caodada/桌面/picture/bird'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片

      #print(images.eval())
      #print(labels.eval())

      #plt.imshow(img)
      #plt.show()
  coord.request_stop()
  coord.join(threads)







