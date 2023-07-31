# V1olet's training script for SHREC23 Sketch-Based 3D Shape Retrieval ANIMAR Challenge.

1. Create a conda environment with name animar and activate it.

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the training script, make sure to change the path to the root folder in the script:
```
bash run.sh
```
- The trained models is saved in the same folder with .h5 extension.
- Note: the original models were trained with batch size 128, but for proof of concept, we trained with batch size 1. You can change the batch size in setting.yaml.