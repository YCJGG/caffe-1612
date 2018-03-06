#
# Pad feature maps to a fixed size that is given by param_str
#
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Reimplement on 2018-03-03

import caffe
import numpy as np
import time

class PadToFixSizeLayer(caffe.Layer):
    """
	Pad feature maps in bottom[0] to the fixed size given in param_str
	top[1] bottom feature map size
    """

    def setup(self, bottom, top):
	# check input/output length, two top and one bottom are accepted
	if len(top) != 2:
	    raise Exception("PadToFixSize only accepts one top")
	if len(bottom) != 1:
	    raise Exception("PadToFixSize accepts one bottom")
	self.targetSize = (int)(self.param_str)

    def reshape(self, bottom, top):
	top_shape = np.array(bottom[0].data.shape)
	top_shape[2] = self.targetSize
	top_shape[3] = self.targetSize
	top[0].reshape(*top_shape)
	top_shape = [1,1,1,2]
	top[1].reshape(*top_shape)

    def forward(self, bottom, top):
	top[0].data[...] = 0
	if bottom[0].data.shape[2] > self.targetSize or bottom[0].data.shape[3] > self.targetSize:
	    raise Exception("Input features are larger that target size.")
	top[0].data[:,:,:bottom[0].data.shape[2],:bottom[0].data.shape[3]] = bottom[0].data[:,:,:,:]
	top[1].data[:] = bottom[0].data.shape[2:]

    def backward(self, top, propagate_down, bottom):
	bottom[0].diff[:,:,:,:] = top[0].diff[:,:,:bottom[0].data.shape[2],:bottom[0].data.shape[3]]
