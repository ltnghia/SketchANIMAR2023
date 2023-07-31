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
from callbacks import *
from utils import *

train_route = 'unzip/shrec23_train_merge_final_dataset'
valid_route = 'unzip/shrec23_test_final_dataset'

setting_cfg = get_settings('setting.yaml')
print(setting_cfg)

os.environ["CUDA_VISIBLE_DEVICES"]=setting_cfg["CUDA_VISIBLE_DEVICES"]

set_memory_growth()

im_size = setting_cfg["im_size"]
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
use_cate_int = True if setting_cfg["label_mode"] == 'cate_int' else False

epochs = setting_cfg["cycle_epoch"] * setting_cfg["n_cycle"]
print('epochs:', epochs)

seedEverything(setting_cfg["seed"])
BATCH_SIZE = setting_cfg["BATCH_SIZE"]
print('BATCH_SIZE:', BATCH_SIZE)

X_train, Y_train, all_class_train = get_data_from_phrase_multiprocessing(train_route)
X_train, Y_train = auto_split_data_fast(X_train, Y_train, all_class_train, valid_ratio=None)
print('train_route:', train_route)

X_valid, Y_valid, all_class_valid = get_data_from_phrase_multiprocessing(valid_route, all_class=all_class_train)
print('valid_route:', valid_route)

train_with_labels = setting_cfg["train_with_labels"]
label_mode = setting_cfg["label_mode"]
train_repeat = setting_cfg["train_repeat"]
train_shuffle = setting_cfg["train_shuffle"]
train_augment = setting_cfg["train_augment"]
im_size_before_crop = setting_cfg["im_size_before_crop"]

valid_with_labels = setting_cfg["valid_with_labels"]
valid_repeat = setting_cfg["valid_repeat"]
valid_shuffle = setting_cfg["valid_shuffle"]
valid_augment = setting_cfg["valid_augment"]

train_n_images = len(Y_train)
train_dataset = build_dataset_from_X_Y(X_train, Y_train, all_class_train, train_with_labels, label_mode, img_size,
                                       BATCH_SIZE, train_repeat, train_shuffle, train_augment, im_size_before_crop)

valid_n_images = len(Y_valid)
valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, all_class_valid, valid_with_labels, label_mode, img_size,
                                       BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment, None)

n_labels = len(all_class_train)

print('n_labels', n_labels)
print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

with open("note.txt", mode='w') as f:
    f.write("n_labels: " + str(n_labels) + "\n")
    f.write("train_n_images: " + str(train_n_images) + "\n")
    f.write("valid_n_images: " + str(valid_n_images) + "\n")

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(n_labels)
    model.summary()

    losses = tf.keras.losses.CategoricalCrossentropy(label_smoothing=setting_cfg['label_smoothing'])

    metrics = ['accuracy']

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=losses,
              metrics=metrics)

if setting_cfg['pretrain_path'] is not None:
    print('Load trained model from:', setting_cfg['pretrain_path'])
    model.load_weights(setting_cfg['pretrain_path'])

base_name = setting_cfg["base_name"]
save_path = f'best_model_sketch_{base_name}_{im_size}_{n_labels}.h5'

monitor = setting_cfg["monitor"]
mode = setting_cfg["mode"]
max_lr = setting_cfg["max_lr"]
min_lr = setting_cfg["min_lr"]
cycle_epoch = setting_cfg["cycle_epoch"]
save_weights_only = setting_cfg["save_weights_only"]
callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch, save_weights_only)

his = model.fit(train_dataset, 
                validation_data=valid_dataset,
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

metric = 'loss'
visual_save_metric(his, metric)

metric = 'accuracy'
visual_save_metric(his, metric)





