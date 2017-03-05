# neuroimage-tensorflow
This Repository contains development framework for using tensorflow in a neuroimaging analyses during BrainHack Boston 2017.

Goal: Build a tensorflow framework capable of working with 3D neuroimaging data (nifti)

## Demo

Setup:
```
mkdir ~/tensorflow-test
cd ~/tensorflow-test
curl -o b40.tar.gz 'https://gate.nmr.mgh.harvard.edu/filedrop2/index.php?p=1m8hsmv9nkj'
tar zxvf b40.tar.gz
cd ./bucker40/
mkdir train
mv 004 ./train
mv 008 ./train
docker run --rm -p 8888:8888 -v ~/tensorflow-test:/notebooks/data gcr.io/tensorflow/tensorflow
```
Go to the URL shown

Inside jupyter:
	- New Terminal
		- `pip install nibabel`

Should now be able to:
```


```

## Sample data

Can be [downloaded here](https://gate.nmr.mgh.harvard.edu/filedrop2/index.php?p=1m8hsmv9nkj).  This link will expire on March 25th, 2017.

## Pre-reqs

- [Tensorflow](https://www.tensorflow.org/install/)
For Python 2.7 CPU support (no GPU support)
```
sudo apt-get install python-pip python-dev
pip install tensorflow
```
- numpy (TODO)
- [nibabel](http://nipy.org/nibabel/)
```
pip install nibabel
```

## Generating a TFrecord
```
python ./genTFrecord.py ./buckner40 'norm' 'aseg' buckner40.tfrecords
```

