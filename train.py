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
IMAGE_PIXELS_3D_SINGLE_CHAN=[255,255,255,1]
NUM_CLASSES=2

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
	
	image  = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
	labels = tf.decode_raw(features['label_raw'], tf.uint8)
	
	#PW 2017/03/03: Zero-center data here?
	image.set_shape([IMAGE_PIXELS])
	image  = tf.reshape(image, IMAGE_PIXELS_3D_SINGLE_CHAN)
	
	labels.set_shape([IMAGE_PIXELS])
	labels  = tf.reshape(image, IMAGE_PIXELS_3D_SINGLE_CHAN)
	
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
	# Convolution layer (https://www.tensorflow.org/api_docs/python/tf/nn/conv3d)
	# tf.nn.conv3d(input, filter, strides, padding, name=None)
    #    input shape: [batch, depth, height, width, in_channels]
	#  	 filter shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
    #    strides shape [1, ?, ?, ?, 1]

   	# Pool layer (https://www.tensorflow.org/api_docs/python/tf/nn/max_pool3d)
	# tf.nn.max_pool3d(input, ksize, strides, padding, name=None)
    #    input shape: [batch, depth, height, width, channels]
	#    ksize: The size of the window for each dimension of the input tensor. 
    #           Must have ksize[0] = ksize[4] = 1
    #    strides shape [1, ?, ?, ?, 1]
 
	print_tensor_shape( images, 'images shape inference' )
	with tf.name_scope('Conv1'):
		W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,1,10],stddev=0.1,dtype=tf.float32),name='W_conv1')
		print_tensor_shape( W_conv1, 'W_conv1 shape')
		conv1_op = tf.nn.conv3d(images, W_conv1, strides=[1,2,2,2,1], padding="SAME", name='conv1_op' )
		print_tensor_shape( conv1_op, 'conv1_op shape')
		relu1_op = tf.nn.relu( conv1_op, name='relu1_op' )
		print_tensor_shape( relu1_op, 'relu1_op shape')
	with tf.name_scope('Pool1'):
		pool1_op = tf.nn.max_pool3d(relu1_op, ksize=[1,3,3,3,1],strides=[1,2,2,2,1], padding='SAME') 
		print_tensor_shape( pool1_op, 'pool1_op shape')
	with tf.name_scope('Conv2'):
		W_conv2 = tf.Variable(tf.truncated_normal([3,3,3,10,100],stddev=0.1,dtype=tf.float32),name='W_conv2')
		print_tensor_shape( W_conv2, 'W_conv2 shape')
		conv2_op = tf.nn.conv3d( pool1_op, W_conv2, strides=[1,2,2,2,1],padding="SAME", name='conv2_op' )
		print_tensor_shape( conv2_op, 'conv2_op shape')
		relu2_op = tf.nn.relu( conv2_op, name='relu2_op' )
		print_tensor_shape( relu2_op, 'relu2_op shape')
	with tf.name_scope('Pool2'):
		pool2_op = tf.nn.max_pool3d(relu2_op, ksize=[1,3,3,3,1],strides=[1,2,2,2,1], padding='SAME')
		print_tensor_shape( pool2_op, 'pool2_op shape')
	with tf.name_scope('Conv3'):
		W_conv3 = tf.Variable(tf.truncated_normal([3,3,3,100,200],stddev=0.1,dtype=tf.float32),name='W_conv3') 
		print_tensor_shape( W_conv3, 'W_conv3 shape')
		conv3_op = tf.nn.conv3d( pool2_op, W_conv3, strides=[1,2,2,2,1],padding='SAME', name='conv3_op' )
		print_tensor_shape( conv3_op, 'conv3_op shape')
		relu3_op = tf.nn.relu( conv3_op, name='relu3_op' )
		print_tensor_shape( relu3_op, 'relu3_op shape')
	with tf.name_scope('Conv4'):
		W_conv4 = tf.Variable(tf.truncated_normal([3,3,3,200,200],stddev=0.1,dtype=tf.float32), name='W_conv4')
		print_tensor_shape( W_conv4, 'W_conv4 shape')
		conv4_op = tf.nn.conv3d( relu3_op, W_conv4, strides=[1,2,2,2,1],padding='SAME', name='conv4_op' )
		print_tensor_shape( conv4_op, 'conv4_op shape')
		relu4_op = tf.nn.relu( conv4_op, name='relu4_op' )
		print_tensor_shape( relu4_op, 'relu4_op shape')
		# optional dropout node.  when set to 1.0 nothing is dropped out
        drop_op = tf.nn.dropout( relu4_op, 1.0 )
        print_tensor_shape( drop_op, 'drop_op shape' )
	# Conv layer to generate the 2 score classes
	with tf.name_scope('Score_classes'):
		W_score_classes = tf.Variable(tf.truncated_normal([1,1,1,200,2],stddev=0.1,dtype=tf.float32),name='W_score_classes')
		print_tensor_shape( W_score_classes, 'W_score_classes_shape')
		score_classes_conv_op = tf.nn.conv3d( drop_op, W_score_classes,strides=[1,1,1,1,1], padding='SAME', name='score_classes_conv_op')
		print_tensor_shape( score_classes_conv_op,'score_conv_op shape')
	# Upscore the results to 256x256x256x2 image
	#  Deconv3d https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose
	#   tf.nn.conv3d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
    #     value: A 5-D Tensor of type float and shape [batch, depth, height, width, in_channels]
    #     filter: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]. filter's in_channels dimension must match that of value.
    #     output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
    #     strides: A list of ints. The stride of the sliding window for each dimension of the input tensor.
	with tf.name_scope('Upscore'):
		W_upscore = tf.Variable(tf.truncated_normal([31,31,31,2,2],stddev=0.1,dtype=tf.float32),name='W_upscore')
		print_tensor_shape( W_upscore, 'W_upscore shape')
		upscore_conv_op = tf.nn.conv3d_transpose( score_classes_conv_op, W_upscore,output_shape=[1,256,256,256,2],strides=[1,16,16,16,1],padding='SAME',name='upscore_conv_op')
        print_tensor_shape(upscore_conv_op, 'upscore_conv_op shape')

	return upscore_conv_op

################################################

with tf.Graph().as_default():
	# Input images and labels.
	images, labels = inputs(train=True, batch_size=2,num_epochs=2)
	print_tensor_shape(images, 'images shape')
	print_tensor_shape(labels, 'labels shape')
	inference(images)
