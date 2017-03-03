# Following along here:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

# Also useful:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py

import tensorflow as tf
import os

INPUT_DIR = './'
TRAIN_FILE = 'b40-train.tfrecords'
VALIDATION_FILE = ''

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified ...
		features={
			'x_dim': tf.FixedLenFeature([], tf.int64),
			'y_dim': tf.FixedLenFeature([], tf.int64),
			'z_dim': tf.FixedLenFeature([], tf.int64),
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label_raw': tf.FixedLenFeature([], tf.string)
		         })

	x_dim  = tf.cast(features['x_dim'], tf.int64)
	y_dim  = tf.cast(features['y_dim'], tf.int64)
	z_dim  = tf.cast(features['z_dim'], tf.int64)

	image  = tf.decode_raw(features['image_raw'], tf.uint8)
	labels = tf.decode_raw(features['label_raw'], tf.uint8)

	image  = tf.reshape(image, tf.stack([x_dim,y_dim,z_dim]))
	labels = tf.reshape(labels, tf.stack([x_dim,y_dim,z_dim]))	

def inputs(train, batch_size, num_epochs):
	"""Reads input data num_epochs times.
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
    filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

	# Even when reading in multiple threads, share the filename queue.
	image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

#inputs(true, 2, 2)
filename = os.path.join(INPUT_DIR,TRAIN_FILE)
with tf.name_scope('input'):
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
image, label = read_and_decode(filename_queue)
