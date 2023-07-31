#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate animar
cd /home/dtpthao/workspace/tf_shrec23_v2-master/

python download.py
python unzip.py
python train.py