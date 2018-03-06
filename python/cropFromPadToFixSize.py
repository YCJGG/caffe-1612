#
# Crop out feature maps from the output of padToFixSize layer.
# The original size of feature maps of bottom[2]
# 
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Reimplement on 2018-03-05

import caffe
import numpy as np
import time

class CropFromPadToFixSizeLayer(caffe.Layer):

    def setup(self, bottom, top):
	# check input/output length, one top and three bottoms are accepted
	if len(top) != 1:
	    raise Exception("CropFromPadToFixSizeLayer only accepts one top")
	if len(bottom) != 3:
	    raise Exception("CropFromPadToFixSizeLayer accepts two bottoms")

    def reshape(self, bottom, top):
	top_shape = np.array(bottom[0].data.shape)
	if top_shape[2] < bottom[2].data.shape[2] or top_shape[3] < bottom[2].data.shape[3]:
	    raise Exception("Input feature cannot be smaller than size of bottom[2]")
	top_shape[2] = bottom[2].data.shape[2]
	top_shape[3] = bottom[2].data.shape[3]
	top[0].reshape(*top_shape)

    def forward(self, bottom, top):
	top[0].data[:,:,:,:] = bottom[0].data[:,:,:bottom[2].data.shape[2],:bottom[2].data.shape[3]]

    def backward(self, top, propagate_down, bottom):
	bottom[0].diff[:,:,:bottom[2].data.shape[2],:bottom[2].data.shape[3]] = top[0].diff[:,:,:,:]
