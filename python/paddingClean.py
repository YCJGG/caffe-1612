#
# Crop parsing results from padded output according to label maps.
# Author: Wei Zhen @ IIE, CAS
# Finish on: 2017-04-17
# Last modified: 2017-04-17

import caffe
import numpy as np
from skimage.transform import resize
import time

class PaddingCleanLayer(caffe.Layer):
    """
	Use bottom[1] label map to obtain paddding information
	Clean parsing results (bottom[0]) in padding area
    """

    def setup(self, bottom, top):
	# check input/output length, one top and two bottoms are accepted
	if len(top) != 1:
	    raise Exception("Padding Clean layer only accepts one top")
	if len(bottom) != 2:
	    raise Exception("Padding Clean layer accepts two bottoms")

    def reshape(self, bottom, top):
	top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
	# 1. copy data from bottom[0] to top[0]
	top[0].data[...] = bottom[0].data.copy()
	# 2. resize label maps bottom[1], and remove data in padding area (with ignore label 255)
	aim_size = np.array(bottom[0].data.shape[2:])
	for batch in range(bottom[1].num):
	    resized_label = resize(bottom[1].data[batch, 0, ...], aim_size, order=0, preserve_range=True)
	    for channel in range(bottom[0].channels):
		top[0].data[batch, channel, resized_label==255] = 0

    def backward(self, top, propagate_down, bottom):
	# 1. copy diff from top[0] to bottom[0]
	bottom[0].diff[...] = top[0].diff.copy()
	# 2. resize label maps bottom[1], and remove data in padding area (with ignore label 255)
	aim_size = np.array(bottom[0].data.shape[2:])
	for batch in range(bottom[1].num):
	    resized_label = resize(bottom[1].data[batch, 0, ...], aim_size, order=0, preserve_range=True)
	    for channel in range(bottom[0].channels):
		bottom[0].diff[batch, channel, resized_label==255] = 0
