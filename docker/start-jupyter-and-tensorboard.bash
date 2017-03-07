#!/usr/bin/env bash

mkdir -p /notebooks/tensorboard-data
/run_jupyter.sh "$@" & 
tensorboard --logdir /notebooks/tensorboard-data


