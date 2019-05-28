import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from PIL import Image
import random

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = int(class_id)
  return labels


def load_train_test_split(dataset_path=''):
  train_images_id = []
  test_images_id = []
  
  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train > 0:
        train_images_id.append(image_id)
      else:
        test_images_id.append(image_id)
        
  return train_images_id, test_images_id

def load_image_path(dataset_path=''):
  image_paths = {}
  with open(os.path.join(dataset_path,'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(dataset_path,'images',pieces[1])
      image_paths[image_id] = path
  return image_paths


def load_bounding_box_annotations(dataset_path=''):
    bboxes = {}

    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = map(int, map(float, pieces[1:]))
            bboxes[image_id] = bbox

    return bboxes

def load_image(dataset_path=''):
  '''image_raw_data = gfile.FastGFile(dataset_path, 'rb').read()
  image = tf.image.decode_jpeg(image_raw_data)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    width = image.shape[0]
    height = image.shape[1]
    #channel = image.shape[2]
    #image = tf.image.resize_images(image, [width, height])
    image_raw = image.tostring()
    return image_raw,width,height'''
  image = Image.open(dataset_path)

  # width = image.size[0]
  # height = image.size[1]
  #
  # if height > width :
  #     ratio = 512 / width
  #     width = 512
  #     height = int(ratio * height)
  # else:
  #     ratio = 512 / height
  #     height = 512
  #     width = int(ratio * width)

  width = 120
  height = 120
  image = image.resize((width,height))
  #plt.imshow()
  #plt.show()

  image_raw = image.tobytes()
  return image_raw,width,height
  
    
labels = load_image_labels(dataset_path=r'E:\dataset\CUB_200_2011')
#print(labels)
train_images_id,test_images_id = load_train_test_split(dataset_path = r'E:\dataset\CUB_200_2011')


#print(train_images_id)
#print(test_images_id)
image_paths = load_image_path(dataset_path=r'E:\dataset\CUB_200_2011')
bboxes  = load_bounding_box_annotations(dataset_path=r'E:\dataset\CUB_200_2011')


''''
val_images_id = random.sample(train_images_id,600)
print(train_images_id)
print(val_images_id)
print(len(train_images_id))
print(len(val_images_id))

for i in val_images_id:
    if i in train_images_id:
        train_images_id.remove(i)

print(len(train_images_id))'''

def create_train_record():
  options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
  options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
  writer = tf.python_io.TFRecordWriter("train.tfrecords")
  
  for image_id in train_images_id:
    train_labels = (labels[image_id])
    file_name = image_paths[image_id]
    train_image_raw,width,height = load_image(dataset_path = file_name)
    example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image_raw])),
            "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))

        })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字
    print(image_id,'train_image processed')
  writer.close()
  print('finish to write data to train.tfrecords file')


def create_val_record():
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter("val.tfrecords")

    for image_id in val_images_id:
        val_labels = (labels[image_id])
        file_name = image_paths[image_id]

        val_image_raw, width, height = load_image(dataset_path=file_name)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[val_labels])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[val_image_raw])),
            "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))

        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字
        print(image_id, 'val_image processed')
    writer.close()
    print('finish to write data to val.tfrecords file')




def create_test_record():

  options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
  options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
  writer = tf.python_io.TFRecordWriter("test.tfrecords")
  for image_id in test_images_id:
    test_labels = (labels[image_id])
    file_name = image_paths[image_id]
    test_image_raw ,width,height= load_image(dataset_path = file_name)
    example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[test_labels])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_image_raw])),
            "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
        })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字
    print(image_id,'test_image processed')
  writer.close()
  print('finish to write data to test.tfrecords file')


train_data = create_train_record()
#val_data = create_val_record()
#test_data = create_test_record()
  
'''
print(labels)
print(train_images[20])
print(train_labels[20])
plt.imshow(train_images[20])
plt.show()
#print(np.array(train_labels))

#for image_id in test_images_id:
  #test_labels.append(labels[image_id])
  #file_name = image_paths[image_id]
  #test_image_value = load_image(sess,dataset_path = file_name)
  #test_images.append(test_image_value)
  #print(image_id,'test_image processed')
'''


#test_run
def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options = options_zlib )
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [200, 200, 3])  #reshape为448*448的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label


def generate_batch(
        example,
        #min_queue_examples,
        batch_size, shuffle):
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
            capacity=20 * batch_size)

    return ret

'''''
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads= tf.train.start_queue_runners(coord=coord)

  image,label = read_and_decode('train.tfrecords')
  threads= tf.train.start_queue_runners(coord=coord)
  for i in range(10):
        example, l = sess.run([image,label])#在会话中取出image和label
        #img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        #example.save('/home/caodada/桌面/picture/bird'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
        plt.imshow(example)
        plt.show()
  coord.request_stop()
  coord.join(threads)'''


''''
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads= tf.train.start_queue_runners(coord=coord)
  image, label = read_and_decode('train.tfrecords')

  #images, labels = generate_batch([image, label], 64, shuffle=True)
  #images = tf.cast(images, dtype=tf.float32)
  #labels = tf.cast(labels, dtype=tf.int64)
  threads = tf.train.start_queue_runners(coord = coord)

  for i in range(500):


      # images, labels = generate_batch([image, label], 10000, 32, shuffle=True)

      example,l = sess.run([image,label])
      print(i,l)

      plt.imshow(example)
      plt.show()


        #img=Image.fromarray(example, 'RGB')#这里Imag e是之前提到的
        #example.save('/home/caodada/桌面/picture/bird'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片

      #print(images.eval())
      #print(labels.eval())

      #plt.imshow(example)
      #plt.show()
  coord.request_stop()
  coord.join(threads)'''
