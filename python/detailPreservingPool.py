#
# Detail-preserving pooling, the lite asym version.
# Note that iter size is not supported.
#
# 	F. Saeedan, N. Weber, M. Goesele, S. Roth
#	 Detail-Preserving Pooling in Deep Networks
#	 In CVPR 2018
# 
# Author: Wei Zhen @ HUST
# Reimplement on 2018-05-02

import caffe
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class detailPreservingPoolLayer(caffe.Layer):

    def setup(self, bottom, top):
	# get parameters
	param_str = self.param_str
	self.kernel_size = (int)(param_str.split(',')[0])
	self.pad = (int)(param_str.split(',')[1])
	self.stride = (int)(param_str.split(',')[2])
	# init alpha and lambda weights
	self.blobs.add_blob(bottom[0].channels)
	self.blobs.add_blob(bottom[0].channels)
	self.blobs[0].data[...] = 0	# alpha
	self.alpha_ = nn.Parameter(torch.from_numpy(self.blobs[0].data[:]).cuda(), requires_grad=True)
	self.blobs[1].data[...] = 0	# lambda
	self.lambda_ = nn.Parameter(torch.from_numpy(self.blobs[1].data[:]).cuda(), requires_grad=True)
	return

    def reshape(self, bottom, top):
	# organize weights
	batch_size = bottom[0].data.shape[0]
	channel = bottom[0].data.shape[1]
	tmp_alpha = torch.stack([self.alpha_.unsqueeze(-1).unsqueeze(-1) for _ in range(batch_size)], dim=0)
	tmp_lambda = torch.stack([self.lambda_.unsqueeze(-1).unsqueeze(-1) for _ in range(batch_size)], dim=0)
	# get input
	self.I = Variable(torch.from_numpy(bottom[0].data[:,:,:,:]).cuda(), requires_grad=True)

	# I_bar
	I_bar = F.upsample_nearest(F.avg_pool2d(self.I, self.kernel_size, stride=self.stride, padding=self.pad), size=self.I.size()[2:])
	# x
	x = F.relu(self.I - I_bar)**2 + 1e-3
	x_bar = F.upsample_nearest(F.avg_pool2d(x, self.kernel_size, stride=self.stride, padding=self.pad), size=x.size()[2:])
	# w
	w = (x/x_bar)**tmp_lambda + tmp_alpha
	w_bar = F.avg_pool2d(w, self.kernel_size, stride=self.stride, padding=self.pad)
	# Iw
	Iw = F.avg_pool2d(self.I * w, self.kernel_size, stride=self.stride, padding=self.pad)
	# output
	self.output = Iw/w_bar

	# reshape top
	top[0].reshape(*self.output.data.cpu().numpy().shape)
	# get output
	top[0].data[...] = self.output.data.cpu().numpy()
	return

    def forward(self, bottom, top):
	return

    def backward(self, top, propagate_down, bottom):
	# backward
	self.output.backward(gradient=torch.from_numpy(top[0].diff[:,:,:,:]).cuda())
	# get diff w.r.t. data and weight
	bottom[0].diff[:,:,:,:] = self.I.grad.data.cpu().numpy()
	self.blobs[0].diff[:] += self.alpha_.grad.data.cpu().numpy()	# accumulate weight's diff
	self.blobs[1].diff[:] += self.lambda_.grad.data.cpu().numpy()
	self.output.detach_()
	# zero grad
	if not self.I.grad is None:
		self.I.grad.detach_()
		self.I.grad.zero_()
	if not self.alpha_.grad is None:
		self.alpha_.grad.detach_()
		self.alpha_.grad.zero_()
	if not self.lambda_.grad is None:
		self.lambda_.grad.detach_()
		self.lambda_.grad.zero_()
	return
