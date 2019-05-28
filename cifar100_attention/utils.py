from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

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


def conv_layer(bottom, name, w_shape, b_shape):
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
    with tf.variable_scope(name+'conv'):
        h = conv_layer(bottom,name=name+'/conv',w_shape=w_shape,b_shape=b_shape)

    with tf.variable_scope(name+'BN'):
        h = batch_normalization_layer(h,axis=axis,phase=phase,name=name+'/BN')

    with tf.variable_scope(name+'ReLU'):
        h = relu_layer(h,name = name+'/ReLU')

    return h



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




# def compatibility_func(L,g):
#     size = L.shape[1]
#     batch = L.shape[0]
#     for i in range(batch):
#         L_vector = tf.reshape(L[i,:,:,:],[size*size,-1]) #(56*56,256)
#         g_vector = tf.expand_dims(g[i,:],axis=0)
#         g_vectorT = tf.transpose(g_vector,(1,0))
#         score_vector = tf.matmul(L_vector,g_vectorT) #(56*56,1)
#         score = tf.nn.softmax(score_vector)
#         img = tf.reshape(score,[size,size])
#         tf.add_to_collection('scoreVector%d'%size,tf.squeeze(score))  #(batch,56*56)
#         tf.add_to_collection('img%d'%size,img)
#     scoreVector = tf.get_collection('scoreVector%d'%size)   #(batch,56*56)
#     image = tf.get_collection('img%d'%size) #(batch,56,56)
#     scoreVector = tf.convert_to_tensor(scoreVector)
#     image = tf.convert_to_tensor(image)
#     print(image.shape)
#
#     return scoreVector,image


# def generate_ga(layer, score_weight):
#     size = layer.shape[1]
#     channel = layer.shape[3]
#     for i in range(channel):
#         ga_image = tf.multiply(layer[:,:,:,i],score_weight)  #(batch,56,56)
#         #gas = tf.reduce_sum(tf.multiply(layer[:,:,:,i],score_weight),[1,2])
#         #print(gas)  #(32,)
#         tf.add_to_collection('gas%d'%size,ga_image)
#         #tf.add_to_collection('ga_img%d'%size,ga_image)
#     ga = tf.get_collection('gas%d'%size)  #(256,batch,56,56)
#     ga = tf.transpose(tf.convert_to_tensor(ga),(1,2,3,0))
#     ga = tf.reduce_mean(ga,[1,2])
#     print(ga.shape)
#     return ga


# def generate_ga(layer, score_vector):
#     batch = layer.shape[0]
#     size = layer.shape[1]
#     channel = layer.shape[3]
#     for i in range(batch):
#         L_vector = tf.reshape(layer[i,:,:,:],[size*size,-1])  #(56*56,256)
#         a_vector = score_vector[i,:]   #(56*56)
#         L_vectorT = tf.transpose(L_vector,(1,0)) #(256,56*56)
#         gas = tf.multiply(L_vectorT,a_vector) #(256,56*56)
#         tf.add_to_collection('gas%d'%size,gas)
#     ga = tf.get_collection('gas%d'%size)  #(batch,256,56*56)
#     ga = tf.convert_to_tensor(ga)
#     ga = tf.reduce_mean(ga,[2])
#     print(ga.shape)
#     return ga



# learn to pay attention
def compatibility_func(L,g):
    size = L.get_shape().as_list()[1]
    batch = L.get_shape().as_list()[0]
    L_vector = tf.reshape(L,[batch,size*size,-1]) #(batch,56*56,256)
    L_vector_T = tf.transpose(L_vector, (0, 2, 1))
    g_vector = tf.expand_dims(g,axis=2)
    score_vector = tf.reduce_sum(tf.multiply(L_vector_T,g_vector),1) #(batch,56*56)
    score = tf.nn.softmax(score_vector) #(batch,56*56)
    # attention map
    imgs = tf.expand_dims(score,axis=1)  #(batch,1,56*56)
    a_score = tf.expand_dims(score,axis=1) #(batch,1,56*56)
    gas = tf.multiply(L_vector_T, a_score)  # (batch,256,56*56)
    ga = tf.reduce_sum(gas,[2])
    print(ga.shape)
    print(imgs.shape)
    return ga,imgs