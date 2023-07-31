#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate animar
cd /home/dtpthao/workspace/shrec23_sketch_animar_infer-master/

python download.py
python unzip.py
python preprocess_data.py