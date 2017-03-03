# Following along here:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

# Also useful:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py

import tensorflow as tf
import os

INPUT_DIR='./'
TRAIN_FILE='b40-train.tfrecords'
VALIDATION_FILE=''
IMAGE_PIXELS=255*255*255
IMAGE_PIXELS_3D=[255,255,255]
NUM_CLASSES=2
CONV_KERNEL_SIZE=5

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def print_tensor_shape(tensor, string):
    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features={'x_dim': tf.FixedLenFeature([], tf.int64),'y_dim': tf.FixedLenFeature([], tf.int64),'z_dim': tf.FixedLenFeature([], tf.int64),'image_raw': tf.FixedLenFeature([], tf.string),'label_raw': tf.FixedLenFeature([], tf.string)})
	
	image  = tf.decode_raw(features['image_raw'], tf.uint8)
	labels = tf.decode_raw(features['label_raw'], tf.uint8)
	
	#PW 2017/03/03: Convert to floats?
	image.set_shape([IMAGE_PIXELS])
	image  = tf.reshape(image, IMAGE_PIXELS_3D)
	
	labels.set_shape([IMAGE_PIXELS])
	labels  = tf.reshape(image, IMAGE_PIXELS_3D)
	
	return image, labels


def inputs(train, batch_size, num_epochs):
	"""
		Reads input data num_epochs times.
		Args:
			train: Selects between the training (True) and validation (False) data.
			batch_size: Number of examples per returned batch.
			num_epochs: Number of times to read the input data, or 0/None to train forever.
		Returns:
			A tuple (images, labels), where:
				* images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
				in the range [-0.5, 0.5].
				* labels is an int32 tensor with shape [batch_size] with the true label,
				a number in the range [0, mnist.NUM_CLASSES).
			Note that an tf.train.QueueRunner is added to the graph, which
			must be run using e.g. tf.train.start_queue_runners().
	"""
	if not num_epochs: num_epochs = None
	filename = os.path.join(INPUT_DIR,TRAIN_FILE if train else VALIDATION_FILE)
	
	with tf.name_scope('input'): 
		filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
	
	# Even when reading in multiple threads, share the filename queue.
	image, label = read_and_decode(filename_queue)
	
	# Shuffle the examples and collect them into batch_size batches.
	# (Internally uses a RandomShuffleQueue.)
	# We run this in two threads to avoid being a bottleneck.
	images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,min_after_dequeue=1000)
	
	return images, sparse_labels


def inference(images):
	#   input: tensor of images
	#   output: tensor of computed logits
    print_tensor_shape( images, 'images shape inference' )
	
	# resize the image tensors to add the number of channels, 1 in this case
	# required to pass the images to various layers upcoming in the graph
	images_re = tf.reshape( images, [-1,256,256,256] ) 
    print_tensor_shape( images, 'images shape inference' )
    
	# Convolution layer
	with tf.name_scope('Conv1'):
	# weight variable 4d tensor, first two dims are patch (kernel) size       
	# third dim is number of input channels and fourth dim is output channels
	# will be convolved with images_re
		W_conv1 = tf.Variable(tf.truncated_normal([CONV_KERNEL_SIZE,CONV_KERNEL_SIZE,CONV_KERNEL_SIZE,10],stddev=0.1,dtype=tf.float32),name='W_conv1')
		print_tensor_shape( W_conv1, 'W_conv1 shape')

# convolution operator.  first arg is the batch of input images with 
# shape [batch, in_height, in_width, in_channels]

# second arg is the filter (weights) with shape 
# [filter_height, filter_width, in_channels, out_channels]

# strides is a 4d tensor.  stride of the sliding window for each
# dimension of input

        conv1_op = tf.nn.conv3d( images_re, W_conv1, 
                     strides=[1,CONV_KERNEL_SIZE,CONV_KERNEL_SIZE,CONV_KERNEL_SIZE], 
                     padding="SAME", name='conv1_op' )
        print_tensor_shape( conv1_op, 'conv1_op shape')

# rectified linear activation function

        relu1_op = tf.nn.relu( conv1_op, name='relu1_op' )
        print_tensor_shape( relu1_op, 'relu1_op shape')

# Pooling layer
    with tf.name_scope('Pool1'):

# max pooling layer
# ksize = size of the window for each input dimension
# strides = stride of the sliding window for each input dimension

        pool1_op = tf.nn.max_pool(relu1_op, ksize=[1,FIXME,FIXME,1],
                                  strides=[1,FIXME,FIXME,1], padding='SAME') 
        print_tensor_shape( pool1_op, 'pool1_op shape')

# Conv layer
    with tf.name_scope('Conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal([FIXME,FIXME,100,200],
                     stddev=0.1,
                     dtype=tf.float32),name='W_conv2')
        print_tensor_shape( W_conv2, 'W_conv2 shape')

        conv2_op = tf.nn.conv2d( pool1_op, W_conv2, 
                     strides=[1,FIXME,FIXME,1],
                     padding="SAME", name='conv2_op' )
        print_tensor_shape( conv2_op, 'conv2_op shape')

        relu2_op = tf.nn.relu( conv2_op, name='relu2_op' )
        print_tensor_shape( relu2_op, 'relu2_op shape')

# Pooling layer
    with tf.name_scope('Pool2'):
        pool2_op = tf.nn.max_pool(relu2_op, ksize=[1,FIXME,FIXME,1],
                                  strides=[1,FIXME,FIXME,1], padding='SAME')
        print_tensor_shape( pool2_op, 'pool2_op shape')
    
# Conv layer
    with tf.name_scope('Conv3'):
        W_conv3 = tf.Variable(tf.truncated_normal([FIXME,FIXME,200,300],
                     stddev=0.1,
                     dtype=tf.float32),name='W_conv3') 
        print_tensor_shape( W_conv3, 'W_conv3 shape')

        conv3_op = tf.nn.conv2d( pool2_op, W_conv3, 
                     strides=[1,FIXME,FIXME,1],
                     padding='SAME', name='conv3_op' )
        print_tensor_shape( conv3_op, 'conv3_op shape')

        relu3_op = tf.nn.relu( conv3_op, name='relu3_op' )
        print_tensor_shape( relu3_op, 'relu3_op shape')
    
# Conv layer
    with tf.name_scope('Conv4'):
        W_conv4 = tf.Variable(tf.truncated_normal([FIXME,FIXME,300,300],
                    stddev=0.1,
                    dtype=tf.float32), name='W_conv4')
        print_tensor_shape( W_conv4, 'W_conv4 shape')

        conv4_op = tf.nn.conv2d( relu3_op, W_conv4, 
                     strides=[1,FIXME,FIXME,1],
                     padding='SAME', name='conv4_op' )
        print_tensor_shape( conv4_op, 'conv4_op shape')

        relu4_op = tf.nn.relu( conv4_op, name='relu4_op' )
        print_tensor_shape( relu4_op, 'relu4_op shape')

# optional dropout node.  when set to 1.0 nothing is dropped out
        drop_op = tf.nn.dropout( relu4_op, 1.0 )
        print_tensor_shape( drop_op, 'drop_op shape' )
    
# Conv layer to generate the 2 score classes
    with tf.name_scope('Score_classes'):
        W_score_classes = tf.Variable(tf.truncated_normal([1,1,300,2],
                            stddev=0.1,dtype=tf.float32),name='W_score_classes')
        print_tensor_shape( W_score_classes, 'W_score_classes_shape')

        score_classes_conv_op = tf.nn.conv2d( drop_op, W_score_classes, 
                       strides=[1,1,1,1], padding='SAME', 
                       name='score_classes_conv_op')
        print_tensor_shape( score_classes_conv_op,'score_conv_op shape')

# Upscore the results to 256x256x2 image
    with tf.name_scope('Upscore'):
        W_upscore = tf.Variable(tf.truncated_normal([31,31,2,2],
                              stddev=0.1,dtype=tf.float32),name='W_upscore')
        print_tensor_shape( W_upscore, 'W_upscore shape')
      
# conv2d_transpose is also referred to in the literature as 
# deconvolution
        upscore_conv_op = tf.nn.conv2d_transpose( score_classes_conv_op, 
                       W_upscore,
                       output_shape=[1,256,256,2],strides=[1,16,16,1],
                       padding='SAME',name='upscore_conv_op')
        print_tensor_shape(upscore_conv_op, 'upscore_conv_op shape')

    return upscore_conv_op


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')



with tf.Graph().as_default():
	# Input images and labels.
	images, labels = inputs(train=True, batch_size=2,num_epochs=2)

###############################
#images, labels = inputs(True, 2, 2)
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#tf.train.start_queue_runners(sess=sess)
#label_val_1, image_val_1 = sess.run([labels, images])
#filename = os.path.join(INPUT_DIR,TRAIN_FILE)
#with tf.name_scope('input'):
#	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
#i, l = read_and_decode(filename_queue)
