# USAGE
#  python ./genTFrecord.py <data-dir> <input-vol-regex> <label-vol-regex>
# EXAMPLE:
#  python ./genTFrecord.py ./buckner40 'norm' 'aseg' buckner40.tfrecords

import numpy as np
import tensorflow as tf
import nibabel as nib
import os
import sys

data_dir = sys.argv[1]
v_regex  = sys.argv[2]
l_regex  = sys.argv[3]
outfile  = sys.argv[4]

print "data_dir:   ", data_dir
print "v_regex:    ", v_regex
print "l_regex:    ", l_regex
print "outfile:    ", outfile

def listfiles(folder):
	for root, folders, files in os.walk(folder):
		for filename in folders + files:
			yield os.path.join(root, filename)

unfiltered_filelist=list(listfiles(data_dir))
input_list = [item for item in unfiltered_filelist if re.search(v_regex,item)]
label_list = [item for item in unfiltered_filelist if re.search(l_regex,item)]

filename_pairs = zip(input_list, label_list)

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# PW 2017/03/01: Can/Should we get away in uint8?
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'buckner40.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
# To compare original to reconstructed images
original_images = []

for v_filename, l_filename in filename_pairs:

	# The volume, in nifti format	
	v_nii = nib.load(v_filename)
	# The volume, in numpy format
	v_np = v_nii.get_data().astype('uint16')
	# The volume, in raw string format
	v_raw = v_np.tostring()

	# The label, in nifti format
	l_nii = nib.load(l_filename)
	# The label, in numpy format
	l_np = l_nii.get_data().astype('uint16')
	# The label, in raw string format
	l_raw = l_np.tostring()

	# Dimensions
	x_dim = v_np.shape[0]
	y_dim = v_np.shape[1]
	z_dim = v_np.shape[2]

    # Put in the original images into array for future check for correctness
	original_images.append((v_np, l_np))

	data_point = tf.train.Example(features=tf.train.Features(feature={
		'x_dim': _int64_feature(x_dim),
		'y_dim': _int64_feature(y_dim),
		'z_dim': _int64_feature(z_dim),
		'image_raw': _bytes_feature(v_raw),
		'label_raw': _bytes_feature(l_raw)}))
    
	writer.write(data_point.SerializeToString())

writer.close()

##############################################################

# Reconstruct images from 'buckner40.tfrecords' can compare to originals

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    
	example = tf.train.Example()
	example.ParseFromString(string_record)
    
	x_dim = int(example.features.feature['x_dim'].int64_list.value[0])    
	y_dim = int(example.features.feature['y_dim'].int64_list.value[0])
	z_dim = int(example.features.feature['z_dim'].int64_list.value[0])    
	image_raw = (example.features.feature['image_raw'].bytes_list.value[0])    
	label_raw = (example.features.feature['label_raw'].bytes_list.value[0])
    
	img_1d = np.fromstring(image_raw, dtype=np.uint16)
	reconstructed_img = img_1d.reshape((x_dim, y_dim, z_dim))
    
	label_1d = np.fromstring(label_raw, dtype=np.uint16)
	reconstructed_label = label_1d.reshape((x_dim, y_dim, z_dim))
    
	reconstructed_images.append((reconstructed_img, reconstructed_label))
 

# Let's check if the reconstructed images match
# the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))











