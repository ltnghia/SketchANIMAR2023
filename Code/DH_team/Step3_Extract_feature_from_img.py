# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import pandas as pd
from tqdm import tqdm


def caculate_feature(img):
    
    # This function generation a vector from a 2D image.
    #Divide the image into a grid of 4x4 squares.
    step = int(224/4)  # 224 is the size of the dimensions of the image.
    img[img>100] = 1 
    totalpoint = np.sum(img) # "Count the white pixels"
    feature = []
    for i in range(4):
        for j in range(4):
            # Calculate the average number of pixels contained in a small square image.
            im = np.sum(img[i*56:i*56+56,j*5:j*56+56])/totalpoint #56=224/4
            x = np.array(im)
            feature.append(x)
            
    #Vector đặc trưng có 16 chiều
    return feature

def main1(src = r'data/3DModel_to_2D',dst = r'data/features_generation_models'):
    modellist = os.listdir(src)
    
    # After being captured from different angles, the 3D models undergo preprocessing, object extraction, and are stored in the 'Mode_Test_bin' folder.
    # This code reads binary images, calculates the feature vectors, and saves them in corresponding CSV files
    # The same process is applied to 2D Sketch files as well
    with tqdm(total=len(modellist)) as pbar:
        for m in modellist:
            _dirpath = os.path.join(src, m)
            print(_dirpath)
            files = os.listdir(_dirpath)
            
            _feature_folder_path = dst
            features = []
    
            if not os.path.exists(_feature_folder_path):
                os.makedirs(_feature_folder_path)
            
            name = os.path.join(_feature_folder_path, m)
            
            if os.path.exists(name):
                continue
            for f in files:    
                filename = os.path.join(_dirpath, f)
                img = cv2.imread(filename)[:,:,0]
                features.append(caculate_feature(img))
        
            f = np.array(features)
            np.savetxt(name + '.csv', f, delimiter=',')    
            pbar.update(1)
            
            
def main2(src = r'data/SketchQuery_Test',dst = r'data/features_generation_sketch'):
    files_2D = os.listdir(src)
    
    # After being captured from different angles, the 3D models undergo preprocessing, object extraction, and are stored in the 'Mode_Test_bin' folder.
    # This code reads binary images, calculates the feature vectors, and saves them in corresponding CSV files
    # The same process is applied to 2D Sketch files as well
    _feature_folder_path = dst
    if not os.path.exists(_feature_folder_path):
        os.makedirs(_feature_folder_path)
    with tqdm(total=len(files_2D)) as pbar:
        for f in files_2D:
            
            features = []
               
            filename = os.path.join(src, f)

            img = cv2.imread(filename)[:,:,0]
            features.append(caculate_feature(img))
        
            ff = np.array(features)
            name = os.path.join(dst, f.split('.')[0]+'.csv')

            np.savetxt(name, ff, delimiter=',')    
            pbar.update(1)

if __name__ == '__main__':
    #Dataset (3Dmodel_to_2D images) Generation vectors
    main1()
    
    #Test Set (Sketch_2D images) Generation vectors
    main2()
    


