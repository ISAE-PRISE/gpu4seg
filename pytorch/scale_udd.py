# ---------------------------------------------------------------------
# GPU4SEG project
# Copyright (C) 2023 ISAE
# 
# Purpose:
# Evaluation of iGPU for semantic segmentation on embedded systems
#
# Contact:
# jean-baptiste.chaudron@isae-supaero.fr
# alfonso.mascarenas-gonzalez@isae-supaero.fr
# ---------------------------------------------------------------------

import os
from PIL import Image
import torchvision.transforms.functional as torch_vision_fn

# TRAINING DATASET ORIGINAL
path_img_dir_train_ori = os.path.join(os.getcwd(), 'UDD6/train/src')
path_lbl_dir_train_ori = os.path.join(os.getcwd(), 'UDD6/train/gt')

# VALIDATION DATASET ORIGINAL
path_img_dir_val_ori = os.path.join(os.getcwd(), 'UDD6/val/src')
path_lbl_dir_val_ori = os.path.join(os.getcwd(), 'UDD6/val/gt')

# TRAINING DATASET SCALED
if not os.path.exists('UDD6_scaled/train/src'):
    os.makedirs('UDD6_scaled/train/src')
path_img_dir_train_scaled = os.path.join(os.getcwd(), 'UDD6_scaled/train/src')
if not os.path.exists('UDD6_scaled/train/gt'):
    os.makedirs('UDD6_scaled/train/gt')
path_lbl_dir_train_scaled = os.path.join(os.getcwd(), 'UDD6_scaled/train/gt')

# VALIDATION DATASET SCALED
if not os.path.exists('UDD6_scaled/val/src'):
    os.makedirs('UDD6_scaled/val/src')
path_img_dir_val_scaled = os.path.join(os.getcwd(), 'UDD6_scaled/val/src')
if not os.path.exists('UDD6_scaled/val/gt'):
    os.makedirs('UDD6_scaled/val/gt')
path_lbl_dir_val_scaled = os.path.join(os.getcwd(), 'UDD6_scaled/val/gt')

list_img_dir_train_ori = [f for f in os.listdir(path_img_dir_train_ori) if not f.startswith('.')]
list_lbl_dir_train_ori = [f for f in os.listdir(path_lbl_dir_train_ori) if not f.startswith('.')]

list_img_dir_val_ori = [f for f in os.listdir(path_img_dir_val_ori) if not f.startswith('.')]
list_lbl_dir_val_ori = [f for f in os.listdir(path_lbl_dir_val_ori) if not f.startswith('.')]

# CHOOSE YOUR OUTPUT SIZES (Rescaled)
Y, X = 360, 480
#Y, X = 360, 640
#Y, X = 480, 820

# SCALE YOUR DATASET IMAGES
i = 0
for img_name in list_img_dir_train_ori:
    img = Image.open(path_img_dir_train_ori + "/" + img_name)
    print("TRAINING SET: Size of the Original image:", img.size)
    img_scaled = torch_vision_fn.resize(img, size=(Y,X),interpolation=torch_vision_fn.InterpolationMode.NEAREST)
    print("TRAINING SET: Size after resize:", img_scaled.size)
    img_scaled.save(path_img_dir_train_scaled + "/" + img_name)

for lbl_name in list_lbl_dir_train_ori:
    lbl = Image.open(path_lbl_dir_train_ori + "/" + lbl_name)
    print("TRAINING SET: Size of the Original label:", lbl.size)
    lbl_scaled = torch_vision_fn.resize(lbl, size=(Y,X),interpolation=torch_vision_fn.InterpolationMode.NEAREST)
    print("TRAINING SET: Size after resize:", lbl_scaled.size)
    lbl_scaled.save(path_lbl_dir_train_scaled + "/" + lbl_name)

for img_name in list_img_dir_val_ori:
    img = Image.open(path_img_dir_val_ori + "/" + img_name)
    print("VALIDATION SET: Size of the Original image:", img.size)
    img_scaled = torch_vision_fn.resize(img, size=(Y,X),interpolation=torch_vision_fn.InterpolationMode.NEAREST)
    print("VALIDATION SET: Size after resize:", img_scaled.size)
    img_scaled.save(path_img_dir_val_scaled + "/" + img_name)

for lbl_name in list_lbl_dir_val_ori:
    lbl = Image.open(path_lbl_dir_val_ori + "/" + lbl_name)
    print("VALIDATION SET: Size of the Original label:", lbl.size)
    lbl_scaled = torch_vision_fn.resize(lbl, size=(Y,X),interpolation=torch_vision_fn.InterpolationMode.NEAREST)
    print("VALIDATION SET: Size after resize:", lbl_scaled.size)
    lbl_scaled.save(path_lbl_dir_val_scaled + "/" + lbl_name)