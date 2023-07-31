import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from custom import *
from dataset import *
from tta import *

os.environ["CUDA_VISIBLE_DEVICES"]=""
SEED = 42
seedEverything(SEED)

# model_paths = [
#     'download/0_3_NN_best_model_sketch_EfficientNetV2S_384_711.h5',
#     'download/0_4_NN_best_model_sketch_ConvNeXtTiny_384_711.h5',
#     'download/0_26_NN_best_model_sketch_EfficientNetV1B5_384_711.h5'
# ]
# 0.43

# model_paths = [
#     'download/0_3_NN_best_model_sketch_EfficientNetV2S_384_711.h5',
#     'download/0_4_NN_best_model_sketch_ConvNeXtTiny_384_711.h5',
#     'download/0_26_NN_best_model_sketch_EfficientNetV1B5_384_711.h5'
# ]
# with flip left right
# 0.4

model_paths = [
    'download/0_3_NN_best_model_sketch_EfficientNetV2S_384_711.h5',
    'download/0_4_NN_best_model_sketch_ConvNeXtTiny_384_711.h5',
    'download/0_26_NN_best_model_sketch_EfficientNetV1B5_384_711.h5',
    'download/last_hit_best_model_sketch_ConvNeXtSmall_384_711.h5'
]
# with flip left right
# 0.46

custom_objects = {
    'resmlp>ChannelAffine' : ChannelAffine,
    'nfnets>ZeroInitGain' : ZeroInitGain,
}

im_size = 384
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
BATCH_SIZE = 1

strategy = auto_select_accelerator()

with strategy.scope():
    models = [tf.keras.models.load_model(path, custom_objects=custom_objects) for path in model_paths]

test_route = 'clean_dataset'

X_test = glob(test_route + '/*.png')

test_n_images = len(X_test)

# augment_func = None
# test_dataset = build_test_dataset(X_test, img_size, BATCH_SIZE, augment_func)

tta_list = [
    None,
    flip_left_right
]

pred_cates = []
for augment_func in tta_list:
    for model in models:
        test_dataset = build_test_dataset(X_test, img_size, BATCH_SIZE, augment_func)
        pred_cate = model.predict(test_dataset, verbose=1)
        pred_cates.append(pred_cate)

pred_cate = np.mean(pred_cates, axis=0)
indices = np.argsort(-pred_cate, axis=1)
indices_name = [[all_classes[i] for i in row] for row in indices]
indices_name = np.array(indices_name)

X_test_name = [os.path.basename(x)[:-4] for x in X_test]

merge_df = np.concatenate((np.expand_dims(X_test_name, axis=1), indices_name), axis=1)

df = pd.DataFrame(merge_df)
df = df.sort_values(by=[0])
df.to_csv('submit.csv', index=False, header=False)
print(df)


