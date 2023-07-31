# SHREC-2023-SKETCH-ANIMAR-main

## Reproduce the result

### Pre-trained phase

Pretrained Model
1. Data Augmentation
By using blender, we employed data augmentation techniques to increase the available training data. By generating 2D sketches from 3D models. Outlines Modifications, Random Deletion, Add Noise.
2. Pretraining
The pretrained model was obtained by training the model on the augmented dataset. This pretraining step helps the model learn basic features and patterns before fine-tuning it on the specific ANIMAR dataset.
3. Fine-tuning
After pretraining, the model was fine-tuned ....
4. Pretrained Weight
You can download the pretrained weight file for the model using the following link: Pretrained Weight Download
5. Source Code and Dataset
The complete source code and dataset for the model are available in separate repositories. Please refer to the following locations for further instructions on running the code and using the dataset:

### Fine-tuning on the training dataset of the competition

For simplicity, we upload the pre-trained weigth into the folder content/original. All preprocessed dataset are also stored in content/original.

To reproduce the result, we run:
``` bash
python main.py
```