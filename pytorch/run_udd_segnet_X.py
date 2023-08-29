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
    print(f"Total Trainable Params SEGNET 1: {total_params}")
    print(f"Total Batch Normalization Params SEGNET 1: {total_params_bn}")
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
def mIoU(pred_mask, mask, smooth=1e-5, n_classes=6):

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
def get_udd_labels_from_rgb(label_numpy_uint8):
	# Here we have seven labels due to aggragated car label
	label = np.zeros(shape=(label_numpy_uint8.shape[0],label_numpy_uint8.shape[1],6), dtype = np.float32)

	#issue here is that the PNGG of UDD is using the PLTE Palette option to encode the color
	#therefore, no RGB channels anymore i.e. vectors but an index of the RGB value in the Palette
	# to show it print(lbl.getpalette())
	other_array = np.array([0, 0, 0],dtype = np.uint8)
	facade_array = np.array([102, 102, 156],dtype = np.uint8)
	road_array = np.array([128, 64, 128],dtype = np.uint8)
	vegetation_array = np.array([107, 142, 35],dtype = np.uint8)
	vehicule_array = np.array([0, 0, 142])
	roof_array = np.array([70, 70, 70],dtype = np.uint8)

	cnt_other = 0
	cnt_facade = 0
	cnt_road = 0
	cnt_vegetation = 0
	cnt_vehicule = 0
	cnt_roof = 0

	for i1 in range(label_numpy_uint8.shape[0]):
		for i2 in range(label_numpy_uint8.shape[1]):
			if (label_numpy_uint8[i1][i2] == 0): # Others Classe 0
				label[i1][i2][0] = 1.0
				cnt_other = cnt_other + 1
			elif (label_numpy_uint8[i1][i2] == 1): # Facade Classe 1
				label[i1][i2][1] = 1.0
				cnt_facade = cnt_facade + 1
			elif (label_numpy_uint8[i1][i2] == 2): # Road Classe 2
				label[i1][i2][2] = 1.0
				cnt_road = cnt_road + 1
			elif (label_numpy_uint8[i1][i2] == 3): # Vegetation Classe 3
				label[i1][i2][3] = 1.0
				cnt_vegetation = cnt_vegetation + 1
			elif  (label_numpy_uint8[i1][i2] == 4): # Vedicule Classe 4
				label[i1][i2][4] = 1.0
				cnt_vehicule = cnt_vehicule + 1
			elif  (label_numpy_uint8[i1][i2] == 5): # Roof Classe 5
				label[i1][i2][5] = 1.0
				cnt_roof = cnt_roof + 1
			else:
				print('WARNING PIXEL UNASSIGNED')
				print(label_numpy_uint8[i1][i2])
	total = cnt_other + cnt_facade + cnt_road + cnt_vegetation + cnt_vehicule + cnt_roof
	print('cnt_other found = ', cnt_other)
	print('cnt_facade found = ', cnt_facade)
	print('cnt_road found = ', cnt_road)
	print('cnt_vegetation found = ', cnt_vegetation)
	print('cnt_vehicule found = ', cnt_vehicule)
	print('cnt_roof found = ', cnt_roof)
	print('total found = ', total)

	return label

#-----------------------------------------------------------------------------------
#----------- MAIN --------------------
#-----------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

in_chanel_size = 3 # RGB image
out_chanel_size = 6 # 6 Classes for UDD
bn_momentum = 0.1 # seams standard in pytorch

# CHANGE MODEL ACCORDING TO YOUR SETTINGS
model = cl_segnet_1(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_2(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_3(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_4(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_5(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_6(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_7(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_8(in_chanel_size, out_chanel_size, bn_momentum)
#model = cl_segnet_9(in_chanel_size, out_chanel_size, bn_momentum)

param_size, param_size_bn = count_parameters(model)
param_size_mb = param_size / 1024**2
param_size_bn_mb = param_size_bn / 1024**2

print('SEGNET TOTAL model size: {:.3f}MB'.format(param_size_mb))
print('SEGNET TOTAL model size with BN: {:.3f}MB'.format(param_size_mb + param_size_bn_mb))

if is_cuda:
    model.cuda()

error = torch_nn.CrossEntropyLoss()

learning_rate = 0.001
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)#, weight_decay=0.0005) # Switch to SGD with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999)) 


# Number epoch and batch size
num_epochs = 300
batch_size = 4

# TRAINING DATASET SCALED
path_img_dir_train_scaled_seq1 = os.path.join(os.getcwd(), 'UDD6_scaled/train/src')
path_lbl_dir_train_scaled_seq1 = os.path.join(os.getcwd(), 'UDD6_scaled/train/gt')
# VALIDATION DATASET SCALED
path_img_dir_val_scaled_seq16 = os.path.join(os.getcwd(), 'UDD6_scaled/val/src')
path_lbl_dir_val_scaled_seq16 = os.path.join(os.getcwd(), 'UDD6_scaled/val/gt')

# LIST in the dataset
list_img_dir_train_scaled_seq1 = [f for f in os.listdir(path_img_dir_train_scaled_seq1) if not f.startswith('.')]
list_lbl_dir_train_scaled_seq1 = [f for f in os.listdir(path_lbl_dir_train_scaled_seq1) if not f.startswith('.')]

list_img_dir_val_scaled_seq16 = [f for f in os.listdir(path_img_dir_val_scaled_seq16) if not f.startswith('.')]
list_lbl_dir_val_scaled_seq16 = [f for f in os.listdir(path_lbl_dir_val_scaled_seq16) if not f.startswith('.')]


# THIS LOOP CREATE THE DATASET OUT OF RGB FILES (TAKES TIME) and CREATE ASSOCIATED NPY FILES
create_npy_label = False # Do you need to create NPY from RGB label
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
			lbl = Image.open(path_lbl_dir_train_scaled_seq1 + "/" + img_name.replace('.JPG', '.png'))
			lbl_numpy_rgb = np.array(lbl, dtype = np.uint8)
			lbl_numpy = get_udd_labels_from_rgb(lbl_numpy_rgb)
			np.save(path_lbl_dir_train_scaled_seq1 + "/" + img_name.replace('.JPG', '') + '.npy',lbl_numpy)
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
			lbl = Image.open(path_lbl_dir_val_scaled_seq16 + "/" + img_name.replace('.JPG', '.png'))
			lbl_numpy_rgb = np.array(lbl, dtype = np.uint8)
			lbl_numpy = get_udd_labels_from_rgb(lbl_numpy_rgb)
			np.save(path_lbl_dir_val_scaled_seq16 + "/" + img_name.replace('.JPG', '') + '.npy',lbl_numpy)
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
			lbl_numpy = np.load(path_lbl_dir_train_scaled_seq1 + "/" + img_name.replace('.JPG', '') + '.npy')
			train_img_list.append(img_numpy)
			train_lbl_list.append(lbl_numpy)
	for img_name in list_img_dir_val_scaled_seq16:
			iteration=iteration+1
			print('Iteration for Val Dataset Creation Image/label number => {}:'.format(iteration))
			img = Image.open(path_img_dir_val_scaled_seq16 + "/" + img_name)
			img_numpy = np.array(img, dtype = np.float32)
			img_numpy /= 255 # Normalization
			lbl_numpy = np.load(path_lbl_dir_val_scaled_seq16 + "/" + img_name.replace('.JPG', '') + '.npy')
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

miou_train_vect_other = []
miou_train_vect_facade = []
miou_train_vect_road = []
miou_train_vect_vegetation = []
miou_train_vect_vehicule = []
miou_train_vect_roof = []

loss_val_vect = []
accur_val_vect = []
miou_val_vect = []

miou_val_vect_other = []
miou_val_vect_facade = []
miou_val_vect_road = []
miou_val_vect_vegetation = []
miou_val_vect_vehicule = []
miou_val_vect_roof = []

mean_iou_val_best = 0

for epoch in range(num_epochs):
	total_loss_train = 0
	total_loss_val = 0
	iter_train = 0
	mean_iou_train = 0
	mean_iou_train_tmp = 0
	mean_iou_train_tmp_vect_tmp = []

	mean_iou_train_other = 0
	mean_iou_train_facade = 0
	mean_iou_train_road = 0
	mean_iou_train_vegetation = 0
	mean_iou_train_vehicule = 0
	mean_iou_train_roof = 0

	iter_iou_train_other = 0
	iter_iou_train_facade = 0
	iter_iou_train_road = 0
	iter_iou_train_vegetation = 0
	iter_iou_train_vehicule = 0
	iter_iou_train_roof = 0

	picc_acc_train = 0
	mean_iou_val = 0
	mean_iou_val_tmp = 0
	mean_iou_val_tmp_vect_tmp = []
	mean_iou_val_other = 0
	mean_iou_val_facade = 0
	mean_iou_val_road = 0
	mean_iou_val_vegetation = 0
	mean_iou_val_vehicule = 0
	mean_iou_val_roof = 0

	iter_iou_val_other = 0
	iter_iou_val_facade = 0
	iter_iou_val_road = 0
	iter_iou_val_vegetation = 0
	iter_iou_val_vehicule = 0
	iter_iou_val_roof = 0

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
			mean_iou_train_other += mean_iou_train_tmp_vect_tmp[0]
			iter_iou_train_other +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[1])):
			mean_iou_train_facade += mean_iou_train_tmp_vect_tmp[1]
			iter_iou_train_facade +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[2])):
			mean_iou_train_road += mean_iou_train_tmp_vect_tmp[2]
			iter_iou_train_road +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[3])):
			mean_iou_train_vegetation += mean_iou_train_tmp_vect_tmp[3]
			iter_iou_train_vegetation +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[4])):
			mean_iou_train_vehicule += mean_iou_train_tmp_vect_tmp[4]
			iter_iou_train_vehicule +=1
		if (not np.isnan(mean_iou_train_tmp_vect_tmp[5])):
			mean_iou_train_roof += mean_iou_train_tmp_vect_tmp[5]
			iter_iou_train_roof +=1
			
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

			model.eval()
			outputs = model(X)
			loss_val = error(outputs, Y)
			total_loss_val += loss_val.item()
			iter_val +=1
			mean_iou_val_tmp, mean_iou_val_tmp_vect_tmp = mIoU(outputs,Y)
			mean_iou_val += mean_iou_val_tmp
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[0])):
				mean_iou_val_other += mean_iou_val_tmp_vect_tmp[0]
				iter_iou_val_other +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[1])):
				mean_iou_val_facade += mean_iou_val_tmp_vect_tmp[1]
				iter_iou_val_facade +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[2])):
				mean_iou_val_road += mean_iou_val_tmp_vect_tmp[2]
				iter_iou_val_road +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[3])):
				mean_iou_val_vegetation += mean_iou_val_tmp_vect_tmp[3]
				iter_iou_val_vegetation +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[4])):
				mean_iou_val_vehicule += mean_iou_val_tmp_vect_tmp[4]
				iter_iou_val_vehicule +=1
			if (not np.isnan(mean_iou_val_tmp_vect_tmp[5])):
				mean_iou_val_roof += mean_iou_val_tmp_vect_tmp[5]
				iter_iou_val_roof +=1
				picc_acc_val += pixel_accuracy(outputs,Y)


	loss_train_vect.append(total_loss_train/(iter_train))
	accur_train_vect.append(picc_acc_train/(iter_train)) 
	miou_train_vect.append(mean_iou_train/(iter_train))

	if (iter_iou_train_other>0):
		miou_train_vect_other.append(mean_iou_train_other/iter_iou_train_other)
	else:
		miou_train_vect_other.append(mean_iou_train_other)

	if (iter_iou_train_facade>0):
		miou_train_vect_facade.append(mean_iou_train_facade/iter_iou_train_facade)
	else:
		miou_train_vect_facade.append(mean_iou_train_facade)

	if (iter_iou_train_road>0):
		miou_train_vect_road.append(mean_iou_train_road/iter_iou_train_road)
	else:
		miou_train_vect_road.append(mean_iou_train_road)

	if (iter_iou_train_vegetation>0):
		miou_train_vect_vegetation.append(mean_iou_train_vegetation/iter_iou_train_vegetation)
	else:
		miou_train_vect_vegetation.append(mean_iou_train_vegetation)

	if (iter_iou_train_vehicule>0):
		miou_train_vect_vehicule.append(mean_iou_train_vehicule/iter_iou_train_vehicule)
	else:
		miou_train_vect_vehicule.append(mean_iou_train_vehicule)

	if (iter_iou_train_roof>0):
		miou_train_vect_roof.append(mean_iou_train_roof/iter_iou_train_roof)
	else:
		miou_train_vect_roof.append(mean_iou_train_roof)

	loss_val_vect.append(total_loss_val/iter_val)
	accur_val_vect.append(picc_acc_val/iter_val) 
	miou_val_vect.append(mean_iou_val/iter_val)

	if (iter_iou_val_other>0):
		miou_val_vect_other.append(mean_iou_val_other/iter_iou_val_other)
	else:
		miou_val_vect_other.append(mean_iou_val_other)

	if (iter_iou_val_facade>0):
		miou_val_vect_facade.append(mean_iou_val_facade/iter_iou_val_facade)
	else:
		miou_val_vect_facade.append(mean_iou_val_facade)

	if (iter_iou_val_road>0):
		miou_val_vect_road.append(mean_iou_val_road/iter_iou_val_road)
	else:
		miou_val_vect_road.append(mean_iou_val_road)

	if (iter_iou_val_vegetation>0):
		miou_val_vect_vegetation.append(mean_iou_val_vegetation/iter_iou_val_vegetation)
	else:
		miou_val_vect_vegetation.append(mean_iou_val_vegetation)

	if (iter_iou_val_vehicule>0):
		miou_val_vect_vehicule.append(mean_iou_val_vehicule/iter_iou_val_vehicule)
	else:
		miou_val_vect_vehicule.append(mean_iou_val_vehicule)

	if (iter_iou_val_roof>0):
		miou_val_vect_roof.append(mean_iou_val_roof/iter_iou_val_roof)
	else:
		miou_val_vect_roof.append(mean_iou_val_roof)

	if ( (mean_iou_val/iter_val) > mean_iou_val_best):
		mean_iou_val_best = (mean_iou_val/iter_val)
		mean_iou_val_best_epoch = epoch
		if (epoch > 5):
			with open("./segnet_X_udd.pth", "wb") as f:
				torch.save(model.state_dict(), f)

	print("Epoch:{}:".format(epoch),
	     " Loss Train :{:.3f}.".format(total_loss_train/(iter_train))," mIoU Train:{:.3f}.".format(mean_iou_train/(iter_train)), " Acc Train:{:.3f}.".format(picc_acc_train/(iter_train))," mIoU Train Vehicule:{:.12f}.".format(mean_iou_train_vehicule/iter_iou_train_vehicule), " mIoU Train Roof:{:.12f}.".format(mean_iou_train_roof/iter_iou_train_roof)," mIoU Train Vegetation:{:.3f}.".format(mean_iou_train_vegetation/iter_iou_train_vegetation))
	print("Epoch:{}:".format(epoch),
	     " Loss Val :{:.3f}.".format(total_loss_val/iter_val)," mIoU Val:{:.3f}.".format(mean_iou_val/iter_val), " Acc Val:{:.3f}.".format(picc_acc_val/iter_val)," mIoU Val Vehicule:{:.12f}.".format(mean_iou_val_vehicule/iter_iou_val_vehicule), " mIoU Val Roof:{:.12f}.".format(mean_iou_val_roof/iter_iou_val_roof)," mIoU Train Vegetation:{:.3f}.".format(mean_iou_val_vegetation/iter_iou_val_vegetation))

		
	print('------')
	print('Average loss @ EPOCH: {}'.format((total_loss_train/(iter_train))))
	print('**********************************************************************************************')
print('BEST MIOU VALIDATION EPOCH =',mean_iou_val_best_epoch)
file = open("./results_segnet_X_udd.csv", mode="w")
file.write("epoch, train_loss, train_accuracy, val_loss, val_accuracy,")
file.write("train_miou_all, train_miou_other, train_miou_facade, train_miou_road,  train_miou_vegetation, train_miou_vehicule, train_miou_roof,")
file.write("val_miou_all, val_miou_other, val_miou_facade, val_miou_road,  val_miou_vegetation, val_miou_vehicule, val_miou_roof\n")
for epoch in range(num_epochs):
	file.write(str(epoch) + "," + str(loss_train_vect[epoch]) + "," + str(accur_train_vect[epoch]) + "," + str(loss_val_vect[epoch]) + "," + str(accur_val_vect[epoch]) + ",")
	file.write(str(miou_train_vect[epoch]) + "," + str(miou_train_vect_other[epoch]) + "," + str(miou_train_vect_facade[epoch]) + "," + str(miou_train_vect_road[epoch]) + ",")
	file.write(str(miou_train_vect_vegetation[epoch]) + "," + str(miou_train_vect_vehicule[epoch]) +  "," + str(miou_train_vect_roof[epoch]) +  ",")
	file.write(str(miou_val_vect[epoch]) + "," + str(miou_val_vect_other[epoch]) + "," + str(miou_val_vect_facade[epoch]) + "," + str(miou_val_vect_road[epoch]) + ",")
	file.write(str(miou_val_vect_vegetation[epoch]) + "," + str(miou_val_vect_vehicule[epoch]) +  "," + str(miou_val_vect_roof[epoch]) + "\n")
file.close()



