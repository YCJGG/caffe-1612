#
# Align the size of one input feature map with the one from another input feature map 
#     using imresize function.
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Finish on: 2016-03-28
# Last modified: 2017-02-18

import caffe
import numpy as np
from skimage.transform import resize
import time

class BlobAlignLayer(caffe.Layer):
    """
	Resize feature maps in bottom[0] in the size of feature maps in bottom[1]
    """

    def setup(self, bottom, top):
	# check input/output length, one top and two bottoms are accepted
	if len(top) != 1:
	    raise Exception("Blob align layer only accepts one top")
	if len(bottom) != 2:
	    raise Exception("Blob align layer accepts two bottoms")
	# select interpolation algorithm
	self.inter_order = 1	# default bilinear
	if self.param_str == "nearest":
	    self.inter_order = 0
	elif self.param_str == "bilinear":
	    self.inter_order = 1
	elif self.param_str == "biquadratic":
	    self.inter_order = 2
	elif self.param_str == "bicubic":
	    self.inter_order = 3
	elif self.param_str == "biquartic":
	    self.inter_order = 4
	elif self.param_str == "biquintic":
	    self.inter_order = 5

    def reshape(self, bottom, top):
	# top/result has the same height and width as the second bottom/input
	#    and the same number and channel as the first bottom/input
	top_shape = np.array(bottom[0].data.shape)
	top_shape[2] = bottom[1].data.shape[2]
	top_shape[3] = bottom[1].data.shape[3]
	top[0].reshape(*top_shape)

    def forward(self, bottom, top):
	#time1 = time.time()
	# bottom[0] provides feature maps to be transformed
	# bottom[1] provides the aim height and width
	# two bottoms can vary in number and channel
	# 1. fetch aim size
	aim_size = np.array(bottom[1].data.shape[2:])
	# 2. resize each feature map using imresize function
	for batch in range(bottom[0].num):
	    for channel in range(bottom[0].channels):
		top[0].data[batch, channel, ...] = resize(bottom[0].data[batch, channel, ...], aim_size, order=self.inter_order, preserve_range=True)

	#print '#########################',time.time()-time1

    def backward(self, top, propagate_down, bottom):
	#time1 = time.time()
	# resize top's diff to the size of bottom[0]
	aim_size = np.array(bottom[0].diff.shape[2:])
	for batch in range(top[0].num):
	    for channel in range(top[0].channels):
		bottom[0].diff[batch, channel, ...] = resize(top[0].diff[batch, channel, ...], aim_size, order=self.inter_order, preserve_range=True)
	# bottom[1].diff = 0
	bottom[1].diff[...] = 0
	#print '!!!!!!!!!!!!!!!!!!!!!!!!!',time.time()-time1
