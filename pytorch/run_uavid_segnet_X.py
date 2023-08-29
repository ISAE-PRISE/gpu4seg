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

from collections import deque
import string
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from PIL import Image
from cl_segnet_1 import cl_segnet_1
from cl_segnet_2 import cl_segnet_2
from cl_segnet_3 import cl_segnet_3
from cl_segnet_4 import cl_segnet_4
from cl_segnet_5 import cl_segnet_5
from cl_segnet_6 import cl_segnet_6
from cl_segnet_7 import cl_segnet_7
from cl_segnet_8 import cl_segnet_8
from cl_segnet_9 import cl_segnet_9
import torchvision.transforms as transforms
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import torchvision.models as models
from prettytable import PrettyTable

#-----------------------------------------------------------------------------------
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    total_params_bn = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
        if ("BN" in name):
            total_params_bn += param
    #print(table)
    print(f"Total Trainable Params SEGNET MODEL: {total_params}")
    print(f"Total Batch Normalization Params SEGNET MODEL: {total_params_bn}")
    return total_params, total_params_bn

#-----------------------------------------------------------------------------------
def pixel_accuracy(pred_mask, mask):
    with torch.no_grad():
        pred_mask = torch.argmax(torch_nn_func.softmax(pred_mask, dim=1), dim=1)
        #pred_mask = torch.argmax(pred_mask, dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(pred_mask, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

#-----------------------------------------------------------------------------------
def mIoU(pred_mask, mask, smooth=1e-5, n_classes=8):

	with torch.no_grad():
		pred_mask = torch.argmax(pred_mask, dim=1)
		pred_mask = pred_mask.contiguous().view(-1)
		mask = torch.argmax(mask, dim=1)
		mask = mask.contiguous().view(-1)

		iou_per_class = []
		for clas in range(0, n_classes): #loop per pixel class
			true_class = (pred_mask == clas)
			true_label = (mask == clas)

			if true_label.long().sum().item() == 0: #no exist label in this loop
				iou_per_class.append(np.nan) #seems that mean iou is one when the class is not in the image
			else:
				intersect = torch.logical_and(true_class, true_label).sum().double().item()
				union = torch.logical_or(true_class, true_label).sum().double().item()
				iou = (intersect + smooth) / (union + smooth)
				iou_per_class.append(iou)

		return np.nanmean(iou_per_class), iou_per_class

#-----------------------------------------------------------------------------------
# Get Label from rgb uavid all labels (including mo)
def get_agg_car_uavid_labels_from_rgb(label_numpy_uint8):
	# Here we have seven labels due to aggragated car label
	label = np.zeros(shape=(label_numpy_uint8.shape[0],label_numpy_uint8.shape[1],7), dtype = np.float32)

	clutter_array = np.array([0, 0, 0],dtype = np.uint8)
	building_array = np.array([128, 0, 0],dtype = np.uint8)
	road_array = np.array([128, 64, 128],dtype = np.uint8)
	static_car_array = np.array([192, 0, 192],dtype = np.uint8)
	tree_array = np.array([0, 128, 0])
	vegetation_array = np.array([128, 128, 0],dtype = np.uint8)
	human_array = np.array([64, 64, 0])
	moving_car_array = np.array([64, 0, 128],dtype = np.uint8)

	cnt_clutter = 0
	cnt_building = 0
	cnt_road = 0
	cnt_car = 0
	cnt_tree = 0
	cnt_vegetation = 0
	cnt_human = 0

	for i1 in range(label_numpy_uint8.shape[0]):
		for i2 in range(label_numpy_uint8.shape[1]):
			if (np.array_equal(label_numpy_uint8[i1][i2],clutter_array)): # Background clutter Classe 0
				label[i1][i2][0] = 1.0
				cnt_clutter = cnt_clutter + 1
			elif (np.array_equal(label_numpy_uint8[i1][i2],building_array)): # Building Classe 1
				label[i1][i2][1] = 1.0
				cnt_building = cnt_building + 1
			elif (np.array_equal(label_numpy_uint8[i1][i2],road_array)): # Road Classe 2
				label[i1][i2][2] = 1.0
				cnt_road = cnt_road + 1
			elif (np.array_equal(label_numpy_uint8[i1][i2],static_car_array) or np.array_equal(label_numpy_uint8[i1][i2],moving_car_array)): # Static_Car Classe 3 and moving car
				label[i1][i2][3] = 1.0
				cnt_car = cnt_car + 1
			elif (np.array_equal(label_numpy_uint8[i1][i2],tree_array)): # Tree Classe 4 
				label[i1][i2][4] = 1.0
				cnt_tree = cnt_tree + 1
			elif (np.array_equal(label_numpy_uint8[i1][i2],vegetation_array)): # Vegetation Classe 5
				label[i1][i2][5] = 1.0
				cnt_vegetation = cnt_vegetation + 1
			elif  (np.array_equal(label_numpy_uint8[i1][i2],human_array)): # Human Classe 6
				label[i1][i2][6] = 1.0
				cnt_human = cnt_human + 1
			else:
				print('WARNING PIXEL UNASSIGNED')
				print(label_numpy_uint8[i1][i2])
	total = cnt_clutter + cnt_building + cnt_road + cnt_car + cnt_tree  + cnt_vegetation + cnt_human
	print('cnt_clutter found = ', cnt_clutter)
	print('cnt_building found = ', cnt_building)
	print('cnt_road found = ', cnt_road)
	print('cnt_car found = ', cnt_car)
	print('cnt_tree found = ', cnt_tree)
	print('cnt_vegetation found = ', cnt_vegetation)
	print('cnt_human found = ', cnt_human)
	print('total found = ', total)

	return label

#-----------------------------------------------------------------------------------
#----------- MAIN --------------------
#-----------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

in_chanel_size = 3 # RGB image

#out_chanel_size = 8 # UAVID = Backgournd (Clutter), Building, Road, Static_Car, Tree, Vegetation, Human, Moving Car
out_chanel_size = 7 # cars and moving car are the same
bn_momentum = 0.1 # seams standard in pytorch

model = cl_segnet_1(in_chanel_size, out_chanel_size, bn_momentum)

param_size, param_size_bn = count_parameters(model)
param_size_mb = param_size / 1024**2
param_size_bn_mb = param_size_bn / 1024**2

print('SEGNET TOTAL model size: {:.3f}MB'.format(param_size_mb))
print('SEGNET TOTAL model size with BN: {:.3f}MB'.format(param_size_mb + param_size_bn_mb))

if is_cuda:
    model.cuda()

error = torch_nn.CrossEntropyLoss()


learning_rate = 0.001
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)#, weight_decay=0.0005) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999)) 

# Number epoch and batch size
num_epochs = 300
batch_size = 4

# TRAINING DATASET SCALED
path_img_dir_train_scaled_seq1 = os.path.join(os.getcwd(), 'uavid_scaled/uavid_train/seq1/Images')
path_lbl_dir_train_scaled_seq1 = os.path.join(os.getcwd(), 'uavid_scaled/uavid_train/seq1/Labels')
# VALIDATION DATASET SCALED
path_img_dir_val_scaled_seq16 = os.path.join(os.getcwd(), 'uavid_scaled/uavid_val/seq16/Images')
path_lbl_dir_val_scaled_seq16 = os.path.join(os.getcwd(), 'uavid_scaled/uavid_val/seq16/Labels')

# LIST in the dataset
list_img_dir_train_scaled_seq1 = [f for f in os.listdir(path_img_dir_train_scaled_seq1) if not f.startswith('.')]
list_lbl_dir_train_scaled_seq1 = [f for f in os.listdir(path_lbl_dir_train_scaled_seq1) if not f.startswith('.')]

list_img_dir_val_scaled_seq16 = [f for f in os.listdir(path_img_dir_val_scaled_seq16) if not f.startswith('.')]
list_lbl_dir_val_scaled_seq16 = [f for f in os.listdir(path_lbl_dir_val_scaled_seq16) if not f.startswith('.')]

create_npy_label = False
# THIS LOOP CREATE THE DATASET OUT OF RGB FILES (TAKES TIME) and CREATE ASSOCIATED NPY FILES
train_img_list = []
train_lbl_list = []
val_img_list = []
val_lbl_list = []
test_img_list = []
iteration = 0
if (create_npy_label):
	for img_name in list_img_dir_train_scaled_seq1:
			iteration=iteration+1
			print('-----------------------------------------------------------------------')
			print('Iteration for Train Dataset Creation Image/label number => {}:'.format(iteration))
			print(img_name)
			img = Image.open(path_img_dir_train_scaled_seq1 + "/" + img_name)
			img_numpy = np.array(img, dtype = np.float32)
			img_numpy /= 255 # Normalization
			lbl = Image.open(path_lbl_dir_train_scaled_seq1 + "/" + img_name)
			lbl_numpy_rgb = np.array(lbl, dtype = np.uint8)
			lbl_numpy = get_agg_car_uavid_labels_from_rgb(lbl_numpy_rgb)
			np.save(path_lbl_dir_train_scaled_seq1 + "/" + img_name.replace('.png', '') + '.npy',lbl_numpy)
			train_img_list.append(img_numpy)
			train_lbl_list.append(lbl_numpy)
	for img_name in list_img_dir_val_scaled_seq16:
			iteration=iteration+1
			print('-----------------------------------------------------------------------')
			print('Iteration for Validation Dataset Creation Image/label number => {}:'.format(iteration))
			print(img_name)
			img = Image.open(path_img_dir_val_scaled_seq16 + "/" + img_name)
			img_numpy = np.array(img, dtype = np.float32)
			img_numpy /= 255 # Normalization
			lbl = Image.open(path_lbl_dir_val_scaled_seq16 + "/" + img_name)
			lbl_numpy_rgb = np.array(lbl, dtype = np.uint8)
			lbl_numpy = get_agg_car_uavid_labels_from_rgb(lbl_numpy_rgb)
			np.save(path_lbl_dir_val_scaled_seq16 + "/" + img_name.replace('.png', '') + '.npy',lbl_numpy)
			val_img_list.append(img_numpy)
			val_lbl_list.append(lbl_numpy)
# THIS LOOP LOAD ASSOCIATED NPY FILES FOR LABEL		
else:
	for img_name in list_img_dir_train_scaled_seq1:
			iteration=iteration+1
			print('Iteration for Train Dataset Creation Image/label number => {}:'.format(iteration))
			img = Image.open(path_img_dir_train_scaled_seq1 + "/" + img_name)
			img_numpy = np.array(img, dtype = np.float32)
			img_numpy /= 255 # Normalization
			lbl_numpy = np.load(path_lbl_dir_train_scaled_seq1 + "/" + img_name.replace('.png', '') + '.npy')
			train_img_list.append(img_numpy)
			train_lbl_list.append(lbl_numpy)
	for img_name in list_img_dir_val_scaled_seq16:
			iteration=iteration+1
			print('Iteration for Val Dataset Creation Image/label number => {}:'.format(iteration))
			img = Image.open(path_img_dir_val_scaled_seq16 + "/" + img_name)
			img_numpy = np.array(img, dtype = np.float32)
			img_numpy /= 255 # Normalization
			lbl_numpy = np.load(path_lbl_dir_val_scaled_seq16 + "/" + img_name.replace('.png', '') + '.npy')
			val_img_list.append(img_numpy)
			val_lbl_list.append(lbl_numpy)

train_img_list_numpy = np.array(train_img_list)
train_lbl_list_numpy = np.array(train_lbl_list)
val_img_list_numpy = np.array(val_img_list)
val_lbl_list_numpy = np.array(val_lbl_list)

train_img_list_torch = torch.from_numpy(train_img_list_numpy)
train_lbl_list_torch = torch.from_numpy(train_lbl_list_numpy)

val_img_list_torch = torch.from_numpy(val_img_list_numpy)
val_lbl_list_torch = torch.from_numpy(val_lbl_list_numpy)

train_img_list_torch = train_img_list_torch.permute(0, 3, 1, 2)
train_lbl_list_torch = train_lbl_list_torch.permute(0, 3, 1, 2)
val_img_list_torch = val_img_list_torch.permute(0, 3, 1, 2)
val_lbl_list_torch = val_lbl_list_torch.permute(0, 3, 1, 2)

#dataset
train_dataset = TensorDataset(train_img_list_torch,train_lbl_list_torch)
val_dataset = TensorDataset(val_img_list_torch,val_lbl_list_torch)

#dataloader
train_loader_shuffled = DataLoader(train_dataset, batch_size, shuffle = True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers=4)

print('---------------------------------------------------------')
print('DATASET IS UP AND READY - RUN SGD')

loss_train_vect = []
accur_train_vect = []
miou_train_vect = []
miou_train_vect_clutter = []
miou_train_vect_building = []
miou_train_vect_road = []
miou_train_vect_car = []
miou_train_vect_tree = []
miou_train_vect_vegetation = []
miou_train_vect_human = []

loss_val_vect = []
accur_val_vect = []
miou_val_vect = []
miou_val_vect_clutter = []
miou_val_vect_building = []
miou_val_vect_road = []
miou_val_vect_car = []
miou_val_vect_tree = []
miou_val_vect_vegetation = []
miou_val_vect_human = []

mean_iou_val_best = 0

for epoch in range(num_epochs):
	total_loss_train = 0
	total_loss_val = 0
	iter_train = 0
	mean_iou_train = 0
	mean_iou_train_tmp = 0
	mean_iou_train_tmp_vect_tmp = []
	mean_iou_train_clutter = 0
	mean_iou_train_building = 0
	mean_iou_train_road = 0
	mean_iou_train_car = 0
	mean_iou_train_tree = 0
	mean_iou_train_vegetation = 0
	mean_iou_train_human = 0
	iter_iou_train_clutter = 0
	iter_iou_train_building = 0
	iter_iou_train_road = 0
	iter_iou_train_car = 0
	iter_iou_train_tree = 0
	iter_iou_train_vegetation = 0
	iter_iou_train_human = 0
	picc_acc_train = 0
	mean_iou_val = 0
	mean_iou_val_tmp = 0
	mean_iou_val_tmp_vect_tmp = []
	mean_iou_val_clutter = 0
	mean_iou_val_building = 0
	mean_iou_val_road = 0
	mean_iou_val_car = 0
	mean_iou_val_tree = 0
	mean_iou_val_vegetation = 0
	mean_iou_val_human = 0
	iter_iou_val_clutter = 0
	iter_iou_val_building = 0
	iter_iou_val_road = 0
	iter_iou_val_car = 0
	iter_iou_val_tree = 0
	iter_iou_val_vegetation = 0
	iter_iou_val_human = 0

	picc_acc_val = 0
	iter_val = 0
	print('**********************************************************************************************')
	print('EPOCH {}:'.format(epoch))
	
	for i, (X, Y) in enumerate(train_loader_shuffled):
		if is_cuda:
			X = X.cuda()
			Y = Y.cuda()

		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()
		model.train()
		outputs = model(X)
		loss_train = error(outputs, Y)
		mean_iou_train_tmp, mean_iou_train_tmp_vect_tmp = mIoU(outputs,Y)
		mean_iou_train += mean_iou_train_tmp
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[0])):
			mean_iou_train_clutter += mean_iou_train_tmp_vect_tmp[0]
			iter_iou_train_clutter +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[1])):
			mean_iou_train_building += mean_iou_train_tmp_vect_tmp[1]
			iter_iou_train_building +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[2])):
			mean_iou_train_road += mean_iou_train_tmp_vect_tmp[2]
			iter_iou_train_road +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[3])):
			mean_iou_train_car += mean_iou_train_tmp_vect_tmp[3]
			iter_iou_train_car +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[4])):
			mean_iou_train_tree += mean_iou_train_tmp_vect_tmp[4]
			iter_iou_train_tree +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[5])):
			mean_iou_train_vegetation += mean_iou_train_tmp_vect_tmp[5]
			iter_iou_train_vegetation +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[6])):
			mean_iou_train_human += mean_iou_train_tmp_vect_tmp[6]
			iter_iou_train_human +=1
			
		picc_acc_train += pixel_accuracy(outputs,Y)
		total_loss_train += loss_train.item()
		iter_train += 1
		# Getting gradients
		loss_train.backward()
		# Updating parameters
		optimizer.step()

		print('Train Loss at {} mini-batch: {}'.format(i, loss_train.item()))
	with torch.no_grad():
		for i, (X, Y) in enumerate(val_loader):
			if is_cuda:
				X = X.cuda()
				Y = Y.cuda()

			# Clear gradients w.r.t. parameters
			model.eval()
			outputs = model(X)
			loss_val = error(outputs, Y)
			total_loss_val += loss_val.item()
			iter_val +=1
			mean_iou_val_tmp, mean_iou_val_tmp_vect_tmp = mIoU(outputs,Y)
			mean_iou_val += mean_iou_val_tmp
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[0])):
				mean_iou_val_clutter += mean_iou_val_tmp_vect_tmp[0]
				iter_iou_val_clutter +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[1])):
				mean_iou_val_building += mean_iou_val_tmp_vect_tmp[1]
				iter_iou_val_building +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[2])):
				mean_iou_val_road += mean_iou_val_tmp_vect_tmp[2]
				iter_iou_val_road +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[3])):
				mean_iou_val_car += mean_iou_val_tmp_vect_tmp[3]
				iter_iou_val_car +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[4])):
				mean_iou_val_tree += mean_iou_val_tmp_vect_tmp[4]
				iter_iou_val_tree +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[5])):
				mean_iou_val_vegetation += mean_iou_val_tmp_vect_tmp[5]
				iter_iou_val_vegetation +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[6])):
				mean_iou_val_human += mean_iou_val_tmp_vect_tmp[6]
				iter_iou_val_human +=1
			picc_acc_val += pixel_accuracy(outputs,Y)


	loss_train_vect.append(total_loss_train/(iter_train))
	accur_train_vect.append(picc_acc_train/(iter_train)) 
	miou_train_vect.append(mean_iou_train/(iter_train))

	if (iter_iou_train_clutter>0):
		miou_train_vect_clutter.append(mean_iou_train_clutter/iter_iou_train_clutter)
	else:
		miou_train_vect_clutter.append(mean_iou_train_clutter)

	if (iter_iou_train_building>0):
		miou_train_vect_building.append(mean_iou_train_building/iter_iou_train_building)
	else:
		miou_train_vect_building.append(mean_iou_train_building)

	if (iter_iou_train_road>0):
		miou_train_vect_road.append(mean_iou_train_road/iter_iou_train_road)
	else:
		miou_train_vect_road.append(mean_iou_train_road)

	if (iter_iou_train_car>0):
		miou_train_vect_car.append(mean_iou_train_car/iter_iou_train_car)
	else:
		miou_train_vect_car.append(mean_iou_train_car)

	if (iter_iou_train_tree>0):
		miou_train_vect_tree.append(mean_iou_train_tree/iter_iou_train_tree)
	else:
		miou_train_vect_tree.append(mean_iou_train_tree)

	if (iter_iou_train_vegetation>0):
		miou_train_vect_vegetation.append(mean_iou_train_vegetation/iter_iou_train_vegetation)
	else:
		miou_train_vect_vegetation.append(mean_iou_train_vegetation)

	if (iter_iou_train_human>0):
		miou_train_vect_human.append(mean_iou_train_human/iter_iou_train_human)
	else:
		miou_train_vect_human.append(mean_iou_train_human)

	loss_val_vect.append(total_loss_val/iter_val)
	accur_val_vect.append(picc_acc_val/iter_val) 
	miou_val_vect.append(mean_iou_val/iter_val)

	if (iter_iou_val_clutter>0):
		miou_val_vect_clutter.append(mean_iou_val_clutter/iter_iou_val_clutter)
	else:
		miou_val_vect_clutter.append(mean_iou_val_clutter)

	if (iter_iou_val_building>0):
		miou_val_vect_building.append(mean_iou_val_building/iter_iou_val_building)
	else:
		miou_val_vect_building.append(mean_iou_val_building)

	if (iter_iou_val_road>0):
		miou_val_vect_road.append(mean_iou_val_road/iter_iou_val_road)
	else:
		miou_val_vect_road.append(mean_iou_val_road)

	if (iter_iou_val_car>0):
		miou_val_vect_car.append(mean_iou_val_car/iter_iou_val_car)
	else:
		miou_val_vect_car.append(mean_iou_val_car)

	if (iter_iou_val_tree>0):
		miou_val_vect_tree.append(mean_iou_val_tree/iter_iou_val_tree)
	else:
		miou_val_vect_tree.append(mean_iou_val_tree)

	if (iter_iou_val_vegetation>0):
		miou_val_vect_vegetation.append(mean_iou_val_vegetation/iter_iou_val_vegetation)
	else:
		miou_val_vect_vegetation.append(mean_iou_val_vegetation)

	if (iter_iou_val_human>0):
		miou_val_vect_human.append(mean_iou_val_human/iter_iou_val_human)
	else:
		miou_val_vect_human.append(mean_iou_val_human)

	if ( (mean_iou_val/iter_val) > mean_iou_val_best):
		mean_iou_val_best = (mean_iou_val/iter_val)
		mean_iou_val_best_epoch = epoch
		if (epoch > 5):
			with open("./segnet_X_uavid.pth", "wb") as f:
				torch.save(model.state_dict(), f)

	print("Epoch:{}:".format(epoch),
	     " Loss Train :{:.3f}.".format(total_loss_train/(iter_train))," mIoU Train:{:.3f}.".format(mean_iou_train/(iter_train)), " Acc Train:{:.3f}.".format(picc_acc_train/(iter_train))," mIoU Train Car:{:.12f}.".format(mean_iou_train_car/iter_iou_train_car), " mIoU Train Human:{:.12f}.".format(mean_iou_train_human/iter_iou_train_human)," mIoU Train Tree:{:.3f}.".format(mean_iou_train_tree/iter_iou_train_tree))
	print("Epoch:{}:".format(epoch),
	     " Loss Val :{:.3f}.".format(total_loss_val/iter_val)," mIoU Val:{:.3f}.".format(mean_iou_val/iter_val), " Acc Val:{:.3f}.".format(picc_acc_val/iter_val)," mIoU Val Car:{:.12f}.".format(mean_iou_val_car/iter_iou_val_car), " mIoU Val Human:{:.12f}.".format(mean_iou_val_human/iter_iou_val_human)," mIoU Train Tree:{:.3f}.".format(mean_iou_val_tree/iter_iou_val_tree))

		
	print('------')
	print('Average loss @ EPOCH: {}'.format((total_loss_train/(iter_train))))
	print('**********************************************************************************************')
print('BEST MIOU VALIDATION EPOCH =',mean_iou_val_best_epoch)
file = open("./results_segnet_X_uavid.csv", mode="w")
file.write("epoch, train_loss, train_accuracy, val_loss, val_accuracy,")
file.write("train_miou_all, train_miou_clutter, train_miou_building, train_miou_road, train_miou_car, train_miou_tree, train_miou_vegetation, train_miou_human,")
file.write("val_miou_all, val_miou_clutter, val_miou_building, val_miou_road, val_miou_car, val_miou_tree, val_miou_vegetation, val_miou_human\n")
for epoch in range(num_epochs):
	file.write(str(epoch) + "," + str(loss_train_vect[epoch]) + "," + str(accur_train_vect[epoch]) + "," + str(loss_val_vect[epoch]) + "," + str(accur_val_vect[epoch]) + ",")
	file.write(str(miou_train_vect[epoch]) + "," + str(miou_train_vect_clutter[epoch]) + "," + str(miou_train_vect_building[epoch]) + "," + str(miou_train_vect_road[epoch]) + "," + str(miou_train_vect_car[epoch]) + ",")
	file.write(str(miou_train_vect_tree[epoch]) + "," + str(miou_train_vect_vegetation[epoch]) + "," + str(miou_train_vect_human[epoch]) +  ",")
	file.write(str(miou_val_vect[epoch]) + "," + str(miou_val_vect_clutter[epoch]) + "," + str(miou_val_vect_building[epoch]) + "," + str(miou_val_vect_road[epoch]) + "," + str(miou_val_vect_car[epoch]) + ",")
	file.write(str(miou_val_vect_tree[epoch]) + "," + str(miou_val_vect_vegetation[epoch]) + "," + str(miou_val_vect_human[epoch]) +   "\n")
file.close()



