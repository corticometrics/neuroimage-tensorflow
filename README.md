# neuroimage-tensorflow
This Repository contains a development framework for using tensorflow for labeling 3D neuroimaging voxel data.  It was developed during BrainHack Boston 2017.  This is a work in progess.  It was based off a 2D pixel labelling example given by Nvidia at a Harvard Compute fest in January 2017.

Goal: Build a tensorflow framework capable of working with 3D neuroimaging data (nifti)

## Demo

Setup:
```
git clone https://github.com/pwighton/neuroimage-tensorflow
cd neuroimage-tensorflow
curl -o b40.tar.gz 'https://gate.nmr.mgh.harvard.edu/filedrop2/index.php?p=1m8hsmv9nkj'
tar zxvf b40.tar.gz
cd ./bucker40/
mkdir train
mv 004 ./train
mv 008 ./train
cd..
```

Build the docker container
```
docker build --no-cache -t tensorflow-tensorboard-nibabel ./docker/
```

Start the docker container (with tensorflow, tensorboard and jupyter), mapping ports for tensorboard and jupyter and mounting the repo dir into `/notebooks/data`
```
docker run -it --rm -p 8888:8888 -p 6006:6006 -v ${PWD}:/notebooks/data tensorflow-tensorboard-nibabel
```

The jupyter URL is shown in the docker terminal window and should look something like
```
http://localhost:8888/?token=bca7c05e6447dc94a80e895bb8e97eb811218e6427af8c12
```

The tensorboard URL is
```
http://localhost:6006/
```

You should now be able to step through the `neuro-example.ipynb` notebook


