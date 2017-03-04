# Creates a .tfrecord file from a directory of nifti images.
#   This assumes your niftis are soreted into subdirs by directory, and a regex
#   can be written to match a volume-filenames and label-filenames
#
# USAGE
#  python ./genTFrecord.py <data-dir> <input-vol-regex> <label-vol-regex>
# EXAMPLE:
#  python ./genTFrecord.py ./buckner40 'norm' 'aseg' buckner40.tfrecords
#
# Based off of this: 
#   http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# imports
import numpy as np
import tensorflow as tf
import nibabel as nib
import os, sys, re

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def select_hipp(x):
	x[x != 17] = 0
	x[x == 17] = 1
	return x

def crop_brain(x):
	x = x[90:130,90:130,90:130] #should take volume zoomed in on hippocampus area
	return x

def preproc_brain(x):
	x = select_hipp(x)
	x = crop_brain(x)   
	return x

def listfiles(folder):
	for root, folders, files in os.walk(folder):
		for filename in folders + files:
			yield os.path.join(root, filename)

def gen_filename_pairs(data_dir, v_re, l_re):
	unfiltered_filelist=list(listfiles(data_dir))
	input_list = [item for item in unfiltered_filelist if re.search(v_regex,item)]
	label_list = [item for item in unfiltered_filelist if re.search(l_regex,item)]
	print("input_list size:    ", len(input_list))
	print("label_list size:    ", len(label_list))
	if len(input_list) != len(label_list):
		print("input_list size and label_list size don't match")
		raise Exception
	return zip(input_list, label_list)

# parse args
data_dir = sys.argv[1]
v_regex  = sys.argv[2]
l_regex  = sys.argv[3]
outfile  = sys.argv[4]
print("data_dir:   ", data_dir)
print("v_regex:    ", v_regex )
print("l_regex:    ", l_regex )
print("outfile:    ", outfile )

# Generate a list of (volume_filename, label_filename) tuples
filename_pairs = gen_filename_pairs(data_dir, v_regex, l_regex)

writer = tf.python_io.TFRecordWriter(outfile)

# To compare original to reconstructed images
original_images = []

for v_filename, l_filename in filename_pairs:

	print("Processing:")
	print("  volume: ", v_filename)
	print("  label:  ", l_filename)	

	# The volume, in nifti format	
	v_nii = nib.load(v_filename)
	# The volume, in numpy format
	v_np = v_nii.get_data().astype('uint16')
	# The volume, in raw string format
	v_np = crop_brain(v_np)
	# The volume, in raw string format
	v_raw = v_np.tostring()

	# The label, in nifti format
	l_nii = nib.load(l_filename)
	# The label, in numpy format
	l_np = l_nii.get_data().astype('uint16')
	# Preprocess the volume
	l_np = preproc_brain(l_np)
	# The label, in raw string format
	l_raw = l_np.tostring()

	# Dimensions
	x_dim = v_np.shape[0]
	y_dim = v_np.shape[1]
	z_dim = v_np.shape[2]
	print("DIMS: " + str(x_dim) + str(y_dim) + str(z_dim))

	# Put in the original images into array for future check for correctness
	# Uncomment to test (this is a memory hog)
	########################################
	# original_images.append((v_np, l_np))

	data_point = tf.train.Example(features=tf.train.Features(feature={
		'image_raw': _bytes_feature(v_raw),
		'label_raw': _bytes_feature(l_raw)}))
    
	writer.write(data_point.SerializeToString())

writer.close()

##############################################################
#  TEST: Reconstruct images from outfile and compare to originals
#  Make sure the line `original_images.append((v_np, l_np))` is uncommented above
##############################################################
#reconstructed_images = []

#record_iterator = tf.python_io.tf_record_iterator(path=outfile)

#for string_record in record_iterator:
    
#	example = tf.train.Example()
#	example.ParseFromString(string_record)
    
#	x_dim = int(example.features.feature['x_dim'].int64_list.value[0])    
#	y_dim = int(example.features.feature['y_dim'].int64_list.value[0])
#	z_dim = int(example.features.feature['z_dim'].int64_list.value[0])    
#	image_raw = (example.features.feature['image_raw'].bytes_list.value[0])    
#	label_raw = (example.features.feature['label_raw'].bytes_list.value[0])
    
#	img_1d = np.fromstring(image_raw, dtype=np.uint16)
#	reconstructed_img = img_1d.reshape((x_dim, y_dim, z_dim))
    
#	label_1d = np.fromstring(label_raw, dtype=np.uint16)
#	reconstructed_label = label_1d.reshape((x_dim, y_dim, z_dim))
    
#	reconstructed_images.append((reconstructed_img, reconstructed_label))

#for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
#    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
#                                                         reconstructed_pair)
#    print(np.allclose(*img_pair_to_compare))
#    print(np.allclose(*annotation_pair_to_compare))
