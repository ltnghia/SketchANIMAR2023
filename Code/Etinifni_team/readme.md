# Etinifni_team

## Pre-trained Phase
1. Data Augmentation
We applied data augmentation techniques using Blender to enhance the available training data. This involved generating 2D sketches from 3D models with modifications such as outline adjustments, random deletion, and noise addition.

2. Pretraining
The pretrained model was trained on the augmented dataset. This pretraining phase enables the model to learn fundamental features and patterns before fine-tuning on the specific ANIMAR dataset.

3. Fine-tuning
After pretraining, the model underwent fine-tuning...

4. Pretrained Weight
Download the pretrained weight file for the model using this [link](https://drive.google.com/file/d/1cOdXMakeJVxrw-gWSieGU6Q5K2mFklSW/view?usp=drive_link).

5. Source Code and Dataset
The complete source code and dataset for the model are available in separate repositories. Please refer to the following locations for detailed instructions on running the code and utilizing the dataset:

## Fine-tuning on the Training Dataset of the Competition
For simplicity, we have uploaded the pre-trained weights to the content/original folder. All preprocessed datasets are also stored there.

To reproduce the results, execute the following command:
```bash
python main.py
```

