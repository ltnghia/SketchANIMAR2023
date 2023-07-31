#The '3DModel_to_2D' folder and 'SketchQuery_Test' contains all the 3D models and Sketch2D images that have been captured from various angles and preprocessed according to the description in the paper.
# SketchQuery_Test.csv, this file lists all Sketch image file names

############################################################################################################################################################
#Step 1 create new env (anaconda env) and install needed library

conda create --name dh python numpy scikit-learn pathlib tqdm opencv pandas

############################################################################################################################################################
# Step 2 
# Change the current working directory to the 'sources' directory.
# Example Our source code is located in D\sources

cd D:
cd D\sources

############################################################################################################################################################
# Step 3
# Execute following command to generate feature vectors for the dataset and the test sketch image set
# all images data located in "data\3DModel_to_2D_to_2D" and "data\SketchQuery_Test"
# If success, 2 folder  will apeard
# If the process is successful, two folders ("features_generation_models" and "features_generation_sketch")
# will be created along with CSV files containing the values of the feature vectors.

python Step3_Extract_feature_from_img.py

############################################################################################################################################################
#Step 4
# execute command 'python Step4_Estimate.py' to generate the 'submission.csv' file that contains the sorted correlation scores between the Sketch and each Model

python Step4_Estimate.py

############################################################################################################################################################