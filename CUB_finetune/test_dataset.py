import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from  PIL import  Image

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
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(dataset_path, 'images', pieces[1])
            image_paths[image_id] = path
    return image_paths


def load_bounding_box_annotations(dataset_path=''):
    bboxes = {}

    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = map(float, pieces[1:])
            bboxes[image_id] = bbox

    return bboxes

def load_image(dataset_path=''):
    '''image_raw_data = gfile.FastGFile(dataset_path, 'rb').read()
    image = tf.image.decode_jpeg(image_raw_data)
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.image.resize_images(image, [200, 200])
      image_raw = image.tobytes()
      return image_raw'''
    image = Image.open(dataset_path)
    image = image.resize((448, 448))
    image_raw = image.tobytes()
    return image_raw


#labels = load_image_labels(dataset_path='C:\\Users\\caodada\\Desktop\\CUB_200_2011')
# print(labels)
#train_images_id, test_images_id = load_train_test_split(dataset_path='C:\\Users\\caodada\\Desktop\\CUB_200_2011')
# print(train_images_id)
# print(test_images_id)
#image_paths = load_image_path(dataset_path='C:\\Users\\caodada\\Desktop\\CUB_200_2011')


def preprocess_for_train(image,image_size):

    distorted_image = tf.image.resize_image_with_crop_or_pad(image,image_size,image_size)
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    return distorted_image

def read_and_decode(filename,flag): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader( )
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'width': tf.FixedLenFeature([],tf.int64),
                                            'height': tf.FixedLenFeature([],tf.int64)})#将image数据和label取出来
    if flag:
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        width = tf.cast(features['width'],tf.int32)
        height = tf.cast(features['height'],tf.int32)
        img = tf.reshape(img,[height,width,3])
        img = tf.random_crop(img, [80, 80, 3])
        img = tf.image.random_flip_left_right(img)
        #bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(img),bounding_boxes=bbox)
        #img = tf.slice(img,bbox_begin,bbox_size)
        #img = tf.reshape(img, [200, 200, 3])  #reshape为448*448的3通道图片
        img = tf.cast(img, tf.float32) * (1. / 255)   #在流中抛出img张量
    else:
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        img = tf.reshape(img, [height, width, 3])
        #img = tf.image.resize_images(img,[120,120] )
        img = tf.image.resize_image_with_crop_or_pad(img, 80, 80)
        img = tf.cast(img, tf.float32) * (1. / 255)
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


''''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(seed=1)
    tf.set_random_seed(np.random.randint(1234))
    #for i in range(3000):
    img_train,label_train = input('C:\\Users\\caodada\\Desktop\\train.tfrecords',datasize=10000,batchsize=32,isShuffel=True)
    img_train,laebl = sess.run([img_train,label_train])
    #img_train = img_train.eval()
        #print(img_train_eval[0].shape())
    print(label_train)
    plt.imshow(img_train[0])
    plt.show()'''

train_images_id, test_images_id = load_train_test_split(dataset_path=r'E:\dataset\CUB_200_2011')
#bboxes  = load_bounding_box_annotations(dataset_path=r'E:\dataset\CUB_200_2011')
#train_bbox = [bboxes[i] for i in range(train_images_id)]
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
