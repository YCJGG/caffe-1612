#
# Try to use pytorch in caffe python layer.
# 
# Author: Wei Zhen @ HUST
# Reimplement on 2018-04-25

import caffe
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class pytorchTestLayer(caffe.Layer):

    def setup(self, bottom, top):
	return

    def reshape(self, bottom, top):
	top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
	self.x = Variable(torch.from_numpy(bottom[0].data[:,:,:,:]), requires_grad=True)
	self.y = F.softmax(self.x, dim=1)
	top[0].data[...] = self.y.data.numpy()

    def backward(self, top, propagate_down, bottom):
	self.y.backward(gradient=Variable(top[0].diff[:,:,:,:]))
	bottom[0].diff[:,:,:,:] = self.x.grad.numpy()
