#
# Gated ave-max pooling with 'per layer' strategy.
# Note that iter size is not supported.
#
# 	C.Y. Lee, P. W. Gallagher, Z. Tu,
#	 Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated and Tree
#	 In AISTA 2016
# 
# Author: Wei Zhen @ HUST
# Reimplement on 2018-04-26

import caffe
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class gatedPoolLayer(caffe.Layer):

    def setup(self, bottom, top):
	# get parameters
	param_str = self.param_str
	self.kernel_size = (int)(param_str.split(',')[0])
	self.pad = (int)(param_str.split(',')[1])
	self.stride = (int)(param_str.split(',')[2])
	# init conv weights
	self.blobs.add_blob(self.kernel_size, self.kernel_size)
	self.blobs[0].data[:,:] = torch.randn(self.kernel_size, self.kernel_size).numpy()
	self.weight = nn.Parameter(torch.from_numpy(self.blobs[0].data[:,:]).cuda(), requires_grad=True)
	return

    def reshape(self, bottom, top):
	# organize weights
	batch_size = bottom[0].data.shape[0]
	channel = bottom[0].data.shape[1]
	tmp_weight = torch.stack([self.weight.unsqueeze(0) for _ in range(channel)], dim=0)
	# get input
	self.x = Variable(torch.from_numpy(bottom[0].data[:,:,:,:]).cuda(), requires_grad=True)
	# get alpha
	alpha = F.conv2d(self.x, tmp_weight, stride=self.stride, padding=self.pad, groups=channel)
	alpha = F.sigmoid(alpha)
	# get pooled results
	self.y = alpha*F.max_pool2d(self.x,self.kernel_size,self.stride,self.pad) + (1-alpha)*F.avg_pool2d(self.x,self.kernel_size,self.stride, self.pad)
	# reshape top
	top[0].reshape(*self.y.data.cpu().numpy().shape)
	# get output
	top[0].data[...] = self.y.data.cpu().numpy()
	return

    def forward(self, bottom, top):
	return

    def backward(self, top, propagate_down, bottom):
	# backward
	self.y.backward(gradient=torch.from_numpy(top[0].diff[:,:,:,:]).cuda())
	# get diff w.r.t. data and weight
	bottom[0].diff[:,:,:,:] = self.x.grad.data.cpu().numpy()
	self.blobs[0].diff[:,:] += self.weight.grad.data.cpu().numpy()	# accumulate weight's diff
	self.y.detach_()
	# zero grad
	if not self.x.grad is None:
		self.x.grad.detach_()
		self.x.grad.zero_()
	if not self.weight.grad is None:
		self.weight.grad.detach_()
		self.weight.grad.zero_()
	return
