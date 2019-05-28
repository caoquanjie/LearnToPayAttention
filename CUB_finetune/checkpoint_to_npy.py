from tensorflow.python import pywrap_tensorflow
import numpy as np
data_path='model.ckpt-130001'
#coding=gbk
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path=data_path #your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

attention_vgg={}
attention_layer = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2',
                 'conv4_3','conv5_1','conv5_2','conv5_3','conv6_1','conv7_1',]
add_info = ['weights','biases']

attention_vgg={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],
        'conv2_2':[[],[]],'conv3_1':[[],[]],'conv3_2':[[],[]],
         'conv3_3':[[],[]],'conv4_1':[[],[]],'conv4_2':[[],[]],
         'conv4_3': [[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]],
         'conv5_3': [[],[]],'conv6_1':[[],[]],'conv7_1':[[],[]]}


for key in var_to_shape_map:
    print ("tensor_name",key)

    str_name = key
    # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    # if str_name.find('Adam') > -1:
    #     continue

    #print('tensor_name:' , str_name)

    if str_name.find('/') > -1:
        names = str_name.split('/')
        #print(names)
        # first layer name and weight, bias
        layer_name = names[1]
        layer_add_info = names[-1]
        #print(layer_name)
        #print(layer_add_info)
    else:
        layer_name = str_name
        layer_add_info = None

    if layer_add_info == 'conv_W':
        attention_vgg[layer_name][0]=reader.get_tensor(key)
    elif layer_add_info == 'conv_b':
        attention_vgg[layer_name][1] = reader.get_tensor(key)
    # else:
    #     attention_vgg[layer_name] = reader.get_tensor(key)

# save npy
# np.save('pretrained_model.npy',attention_vgg)
# print('save npy over...')
print(attention_vgg['conv4_3'][0].shape)