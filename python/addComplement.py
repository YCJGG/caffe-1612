#
# To create an additional channel for single channel input so that:
#	channel1 + channel2 = int(param_str)
# Used in regression output + CRF model
# 
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Reimplement on 2018-03-03

import caffe
import numpy as np
import time

class AddComplementLayer(caffe.Layer):

    def setup(self, bottom, top):
	# check input/output length, one top and one bottoms are accepted
	if len(top) != 1:
	    raise Exception("AddComplementLayer only accepts one top")
	if len(bottom) != 1:
	    raise Exception("AddComplementLayer accepts one bottom")
	self.totalValue = (float)(self.param_str)

    def reshape(self, bottom, top):
	top_shape = np.array(bottom[0].data.shape)
	if top_shape[1] != 1:
	    raise Exception("Input feature must have only one channel")
	top_shape[1] = 2
	top[0].reshape(*top_shape)

    def forward(self, bottom, top):
	top[0].data[:,0,:,:] = bottom[0].data[:,:,:,:]
	top[0].data[:,1,:,:] = self.totalValue - bottom[0].data[:,:,:,:]

    def backward(self, top, propagate_down, bottom):
	bottom[0].diff[:,:,:,:] = top[0].diff[:,0,:,:]
