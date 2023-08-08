# DH Team
The 'data/3DModel_to_2D/' folder and 'data/ketchQuery_Test/' contain all the 3D models and Sketch2D images that have been captured from various angles and preprocessed according to the description in the paper.

`SketchQuery_Test.csv` contains a list of all Sketch image file names.

Please consider download the whole zip file from here: [link](https://drive.google.com/file/d/13UnyuAhK7u-HZFBKf53I5YXE3pKvy1A1/view?usp=drive_link)

---

## Step 1: Create a New Environment (Anaconda Environment) and Install Required Libraries
Create a new environment and install the required libraries using the following commands:
```bash
conda create --name dh python=3.9
conda activate dh
pip install -r requirements.txt
```

## Step 2: Generate Feature Vectors
Execute the following command to generate feature vectors for the dataset and the test sketch image set. All image data is located in "data\3DModel_to_2D_to_2D" and "data\SketchQuery_Test". If successful, two folders ("features_generation_models" and "features_generation_sketch") will be created along with CSV files containing the values of the feature vectors.
```bash
python Step2_Extract_feature_from_img.py
```

## Step 3: Generate Correlation Scores
Execute the command python Step4_Estimate.py to generate the submission.csv file that contains the sorted correlation scores between the Sketch and each Model.
```bash
python Step3_Estimate.py
```