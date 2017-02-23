#
# A prob layer for debug.
# Author: Wei Zhen @ IIE, CAS
# Finish on: 2017-02-11
# Last modified: 2017-02-11

import caffe
import numpy as np
from skimage.transform import resize
import time
import matplotlib.pyplot as plt

class ProbLayer(caffe.Layer):
    """
	Resize feature maps in bottom[0] in the size of feature maps in bottom[1]
    """

    def setup(self, bottom, top):
	return

    def reshape(self, bottom, top):
	#top[0].reshape(*bottom[0].shape)
	return

    def forward(self, bottom, top):
	#top[0].data[...] = bottom[0].data[...]
	#----------- visualize -------------
	#print bottom[1].data.shape
	#plt.subplot(1,2,1)
	#plt.imshow(bottom[0].data[0,:,:,:].transpose([1,2,0]))
	print bottom[0].data.shape
	plt.imshow(bottom[0].data[0,0,:,:])
	#plt.subplot(1,2,2)
	#plt.imshow(bottom[1].data[0,0,:,:])
	plt.show()

	return

    def backward(self, top, propagate_down, bottom):
	# --------- check gradient ----------
	#print top[0].diff.sum()	
	#bottom[0].diff[...] = top[0].diff[...]

	return
