import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt



def read_cifar10(data_dir, is_train):
    """Read CIFAR10

    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32

    """
    image_size = 32
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width * img_height * img_depth

    with tf.name_scope('input'):

        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % ii)
                         for ii in np.arange(1, 6)]
            filename_queue = tf.train.string_input_producer(filenames)

            reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

            key, value = reader.read(filename_queue)

            record_bytes = tf.decode_raw(value, tf.uint8)

            label = tf.slice(record_bytes, [0], [label_bytes])
            label = tf.cast(label, tf.int64)
            label = tf.squeeze(label)

            image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
            image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
            image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
            image = tf.image.resize_image_with_crop_or_pad(
                image, image_size + 4, image_size + 4)
            image = tf.random_crop(image, [image_size, image_size, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) * (1. / 255)
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

            filename_queue = tf.train.string_input_producer(filenames)

            reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

            key, value = reader.read(filename_queue)

            record_bytes = tf.decode_raw(value, tf.uint8)

            label = tf.slice(record_bytes, [0], [label_bytes])
            label = tf.cast(label, tf.int64)
            label = tf.squeeze(label)

            image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
            image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
            image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
            image = tf.image.resize_image_with_crop_or_pad(
                image, image_size, image_size)
            image = tf.cast(image, tf.float32) * (1. / 255)
            #image = tf.image.per_image_standardization(image)  # substract off the mean and divide by the variance
        return image,label

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


# ''''
# with tf.Session() as sess:
#   coord = tf.train.Coordinator()
#   threads= tf.train.start_queue_runners(coord=coord)
#   train_image, train_label = read_cifar10(data_dir='C:\\Users\\caodada\\Desktop\\cifar-10-batches-bin\\', is_train=True,
#                                             batch_size=32, shuffle=True)
#
#   train_images,train_labels = generate_batch([train_image,train_label],batch_size= 32,shuffle= True)
#
#   #image, label = read_and_decode('C:\\Users\\caodada\\Desktop\\train.tfrecords')
#
#   #images, labels = generate_batch([image, label], 64, shuffle=True)
#   #train_images = tf.cast(train_images, dtype=tf.float32)
#   #train_labels = tf.cast(train_labels, dtype=tf.int64)
#   threads = tf.train.start_queue_runners(coord = coord)
#
#
#   for i in range(200):
#
#       example,l = sess.run([train_images,train_labels])
#       print(i,l)
#
#       plt.imshow(example[0])
#       plt.show()
#   coord.request_stop()
#   coord.join(threads)'''


# image, label = read_cifar10(r'C:\Users\caodada\Desktop\cifar10_att3_92.3%\cifar-10-batches-bin',
#                                  is_train = True)
# images,labels = generate_batch([image,label],batch_size=32,shuffle=True)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(50):
#
#         example,l = sess.run([images,labels])
#         print(i,l)
#         print(example[0].shape)
#         plt.imshow(example[0])
#         plt.show()
#
#     coord.request_stop()
#     coord.join(threads)
