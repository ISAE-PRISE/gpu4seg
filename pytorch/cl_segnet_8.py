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

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import os

is_cuda = torch.cuda.is_available()

class cl_segnet_8(torch_nn.Module):

	def __init__(self, in_chanel_size, out_chanel_size, bn_momentum):
		super(cl_segnet_8, self).__init__()

		# Original SegNet Architecture
		# https://arxiv.org/pdf/1511.00561.pdf
		self.in_chn = in_chanel_size
		self.out_chn = out_chanel_size

		self.chn_1 = 64
		self.chn_2 = 128
		self.chn_3 = 256

		# ENCODER 
		self.ConvEn11 = torch_nn.Conv2d(self.in_chn, self.chn_1, kernel_size=3, padding=1, bias=False)
		self.BNEn11 = torch_nn.BatchNorm2d(self.chn_1, momentum=bn_momentum)
		self.ConvEn12 = torch_nn.Conv2d(self.chn_1, self.chn_1, kernel_size=3, padding=1, bias=False)
		self.BNEn12 = torch_nn.BatchNorm2d(self.chn_1, momentum=bn_momentum)

		self.ConvEn21 = torch_nn.Conv2d(self.chn_1, self.chn_2, kernel_size=3, padding=1, bias=False)
		self.BNEn21 = torch_nn.BatchNorm2d(self.chn_2, momentum=bn_momentum)
		self.ConvEn22 = torch_nn.Conv2d(self.chn_2, self.chn_2, kernel_size=3, padding=1, bias=False)
		self.BNEn22 = torch_nn.BatchNorm2d(self.chn_2, momentum=bn_momentum)

		self.ConvEn31 = torch_nn.Conv2d(self.chn_2, self.chn_3, kernel_size=3, padding=1, bias=False)
		self.BNEn31 = torch_nn.BatchNorm2d(self.chn_3, momentum=bn_momentum)
		self.ConvEn32 = torch_nn.Conv2d(self.chn_3, self.chn_3, kernel_size=3, padding=1, bias=False)
		self.BNEn32 = torch_nn.BatchNorm2d(self.chn_3, momentum=bn_momentum)


		# DECODER
		# Each stage corresponds to their respective counterparts in ENCODER
		self.ConvDe32 = torch_nn.Conv2d(self.chn_3, self.chn_3, kernel_size=3, padding=1, bias=False)
		self.BNDe32 = torch_nn.BatchNorm2d(self.chn_3, momentum=bn_momentum)
		self.ConvDe31 = torch_nn.Conv2d(self.chn_3, self.chn_2, kernel_size=3, padding=1, bias=False)
		self.BNDe31 = torch_nn.BatchNorm2d(self.chn_2, momentum=bn_momentum)

		self.ConvDe22 = torch_nn.Conv2d(self.chn_2, self.chn_2, kernel_size=3, padding=1, bias=False)
		self.BNDe22 = torch_nn.BatchNorm2d(self.chn_2, momentum=bn_momentum)
		self.ConvDe21 = torch_nn.Conv2d(self.chn_2, self.chn_1, kernel_size=3, padding=1, bias=False)
		self.BNDe21 = torch_nn.BatchNorm2d(self.chn_1, momentum=bn_momentum)

		self.ConvDe12 = torch_nn.Conv2d(self.chn_1, self.chn_1, kernel_size=3, padding=1, bias=False)
		self.BNDe12 = torch_nn.BatchNorm2d(self.chn_1, momentum=bn_momentum)
		self.ConvDe11 = torch_nn.Conv2d(self.chn_1, self.out_chn, kernel_size=3, padding=1, bias=False)
		self.BNDe11 = torch_nn.BatchNorm2d(self.out_chn, momentum=bn_momentum)

		# Weight initialization Encoders
		torch.nn.init.kaiming_uniform_(self.ConvEn11.weight)
		torch.nn.init.kaiming_uniform_(self.ConvEn12.weight)
		torch.nn.init.kaiming_uniform_(self.ConvEn21.weight)
		torch.nn.init.kaiming_uniform_(self.ConvEn22.weight)
		torch.nn.init.kaiming_uniform_(self.ConvEn31.weight)
		torch.nn.init.kaiming_uniform_(self.ConvEn32.weight)
		# Weight initialization Decoders
		torch.nn.init.kaiming_uniform_(self.ConvDe32.weight)
		torch.nn.init.kaiming_uniform_(self.ConvDe31.weight)
		torch.nn.init.kaiming_uniform_(self.ConvDe22.weight)
		torch.nn.init.kaiming_uniform_(self.ConvDe21.weight)
		torch.nn.init.kaiming_uniform_(self.ConvDe12.weight)
		torch.nn.init.kaiming_uniform_(self.ConvDe11.weight)

	def forward(self, x):

		#ENCODE LAYERS
		#Stage 1
		x = torch_nn_func.relu(self.BNEn11(self.ConvEn11(x))) 
		x = torch_nn_func.relu(self.BNEn12(self.ConvEn12(x))) 
		x, ind1 = torch_nn_func.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
		size1 = x.size()

		#Stage 2
		x = torch_nn_func.relu(self.BNEn21(self.ConvEn21(x))) 
		x = torch_nn_func.relu(self.BNEn22(self.ConvEn22(x))) 
		x, ind2 = torch_nn_func.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
		size2 = x.size()

		#Stage 3
		x = torch_nn_func.relu(self.BNEn31(self.ConvEn31(x))) 
		x = torch_nn_func.relu(self.BNEn32(self.ConvEn32(x))) 	
		x, ind3 = torch_nn_func.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)

		#DECODE LAYERS
		#Stage 3
		x = torch_nn_func.max_unpool2d(x, ind3, kernel_size=2, stride=2, output_size=size2)
		x = torch_nn_func.relu(self.BNDe32(self.ConvDe32(x)))
		x = torch_nn_func.relu(self.BNDe31(self.ConvDe31(x)))

		#Stage 2
		x = torch_nn_func.max_unpool2d(x, ind2, kernel_size=2, stride=2, output_size=size1)
		x = torch_nn_func.relu(self.BNDe22(self.ConvDe22(x)))
		x = torch_nn_func.relu(self.BNDe21(self.ConvDe21(x)))

		#Stage 1
		x = torch_nn_func.max_unpool2d(x, ind1, kernel_size=2, stride=2)
		x = torch_nn_func.relu(self.BNDe12(self.ConvDe12(x)))
		x = torch_nn_func.relu(self.BNDe11(self.ConvDe11(x)))

		return x
