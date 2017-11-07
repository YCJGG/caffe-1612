#
# Perform grabcut as post-processing for parsing task.
# Author: Wei Zhen @ CS, HUST
# Finish on: 2017-11-07

import caffe
import numpy as np
from skimage.transform import resize
import time
import matplotlib.pyplot as plt
import cv2

class grabCutLayer(caffe.Layer):
    """
	Perform 2-class grabcut on each channel of parsing results.
	No backward operations.
	bottom[0]: parsing results, normalized by SoftMax (NxCxHxW)
	bottom[1]: image / data (Nx3xHxW)
	top[0]: processed results (NxCxHxW)
    """

    def setup(self, bottom, top):
	self.bgModel = np.zeros((1,65),np.float64)
	self.fgModel = np.zeros((1,65),np.float64)

    def reshape(self, bottom, top):
	top[0].reshape(*bottom[0].shape)
	# reverse data and get original image
	img = np.zeros(tuple(bottom[1].shape))
	img = bottom[1].data[...]
	img[:,0,...] += 104.008
	img[:,1,...] += 116.669
	img[:,2,...] += 122.675
	img = img[:,::-1,:,:]
	img[img>255] = 255
	self.img = np.zeros((bottom[1].shape[0],bottom[1].shape[2],bottom[1].shape[3],bottom[1].shape[1]))
	for n in range(bottom[0].num):
		self.img[n,:,:,:] = img[n,:,:,:].transpose(1,2,0).astype(np.uint8)

    def forward(self, bottom, top):
	for n in range(bottom[0].num):
		for channel in bottom[0].data[n,:,:,:]:
			channel[channel>0.5] = cv2.GC_FGD
			channel[channel!=cv2.GC_FGD] = cv2.GC_BGD
			if channel.sum() == 0:
	    			continue
			# loop 5 times
			cv2.grabCut(np.array(self.img[n,:,:,:], dtype=np.uint8), channel.astype(np.uint8), None, self.bgModel, self.fgModel, 5, cv2.GC_INIT_WITH_MASK)
			channel = np.where((channel==2)|(channel==0),0,1).astype(np.uint8)

    def backward(self, top, propagate_down, bottom):
	# Not Implemented
	return
