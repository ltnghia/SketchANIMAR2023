import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from dataset import *
from model import *
from losses import *
from utils import *

weight_path = 'best_model_sketch_EfficientNetV2S_384_711.h5'
n_labels = 711

setting_cfg = get_settings('setting.yaml')
print(setting_cfg)

os.environ["CUDA_VISIBLE_DEVICES"]=""

set_memory_growth()

im_size = setting_cfg["im_size"]
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
use_cate_int = True if setting_cfg["label_mode"] == 'cate_int' else False

strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(n_labels)
    model.load_weights(weight_path)

    # model = tf.keras.models.load_model(weight_path)
    
    model.summary()

test_route = 'unzip/shrec23_test_predict'

im_size = setting_cfg["im_size"]
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
use_cate_int = True if setting_cfg["label_mode"] == 'cate_int' else False

seedEverything(setting_cfg["seed"])
BATCH_SIZE = setting_cfg["BATCH_SIZE"]
print('BATCH_SIZE:', BATCH_SIZE)

X_test = glob(test_route + '/*.png')

test_n_images = len(X_test)
test_dataset = build_dataset_from_X_Y(X_test, None, None, False, None, img_size,
                                      BATCH_SIZE, False, False, False, None)

try:
    pred_cate, pred_emb = model.predict(test_dataset, verbose=1)
except:
    pred_cate = model.predict(test_dataset, verbose=1)

train_route = 'unzip/shrec23_train_merge_final_dataset'
all_class = sorted(os.listdir(train_route))

indices = np.argsort(-pred_cate, axis=1)
indices_name = [[all_class[i] for i in row] for row in indices]
indices_name = np.array(indices_name)
print(indices_name.shape)

X_test_name = [os.path.basename(x)[:-4] for x in X_test]

merge_df = np.concatenate((np.expand_dims(X_test_name, axis=1), indices_name), axis=1)

df = pd.DataFrame(merge_df)
df = df.sort_values(by=[0])
df.to_csv('submit.csv', index=False, header=False)
print(df)


