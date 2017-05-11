#
# Align the size of one input feature map with the one from another input feature map 
#     using imresize function.
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Finish on: 2016-03-28
# Last modified: 2017-04-24

import caffe
import numpy as np
from skimage.transform import resize
from multiprocessing import Pool as ThreadPool

# global function
def run_resize(map_arg):
    return resize(map_arg[0], map_arg[1], preserve_range=True)

class ImgResizeLayer(caffe.Layer):
    """
	Resize feature maps in bottom[0] in the size of param_str
    """

    def setup(self, bottom, top):
	# check input/output length, one top and two bottoms are accepted
	if len(top) != 1:
	    raise Exception("Blob align layer only accepts one top")
	if len(bottom) != 1:
	    raise Exception("Blob align layer accepts one bottom")
	self.newshape = int(self.param_str)
	# open multiprocessing pool
	self.pool = ThreadPool(8)

    def reshape(self, bottom, top):
	# top/result has the same height and width as the second bottom/input
	#    and the same number and channel as the first bottom/input
	top_shape = np.array((bottom[0].num, bottom[0].channels, self.newshape, self.newshape))
	top[0].reshape(*top_shape)

    def forward(self, bottom, top):
	# bottom[0] provides feature maps to be transformed
	# 1. fetch aim size
	aim_size = np.array([self.newshape, self.newshape])
	# 2. resize each feature map using imresize function
	for batch in range(bottom[0].num):
	    map_arg = []
	    for channel in range(bottom[0].channels):
		map_arg.append([bottom[0].data[batch,channel,...], aim_size])
	    top[0].data[batch,...] = self.pool.map(run_resize, map_arg)

    def backward(self, top, propagate_down, bottom):
	aim_size = np.array(bottom[0].diff.shape[2:])
	for batch in range(top[0].num):
	    map_arg = []
	    for channel in range(top[0].channels):
		map_arg.append([top[0].diff[batch,channel,...], aim_size])
	    bottom[0].diff[batch, ...] = self.pool.map(run_resize, map_arg)
	return
