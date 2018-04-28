#
# Horizontally flip face segmentation label maps if ' mirror' is applied in data layer
# Author: Wei Zhen @ HUST
# Finish on: 2017-10-25

import caffe
import numpy as np
import matplotlib.pyplot as plt

class faceSegLabelMirrorLayer(caffe.Layer):

    def setup(self, bottom, top):
	# check input/output length, one top and two bottoms are accepted
	if len(top) != 1:
	    raise Exception("Face seg label mirror layer only accepts one top")
	if len(bottom) != 1:
	    raise Exception("Face seg label mirror layer accepts one bottom")

    def reshape(self, bottom, top):
	top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
	top[0].data[...] = bottom[0].data[...]
	# bottom[0] is seg label map
	# 1. check whether 'mirror' is applied in data layer
	#   by checking the label value at right pupil
	#   pupil location: [85,172]
	#   right eye label: 8, left eye label: 7
	#   right eye brow label: 10, left eye brow label: 9
	if bottom[0].data[0,0,85,172] == 7:
	    # 2. exchange the label values of eyes and eye brows
	    top[0].data[0,0,top[0].data[0,0,:,:]==7] = 100
	    top[0].data[0,0,top[0].data[0,0,:,:]==8] = 7
	    top[0].data[0,0,top[0].data[0,0,:,:]==100] = 8

	    top[0].data[0,0,top[0].data[0,0,:,:]==9] = 100
	    top[0].data[0,0,top[0].data[0,0,:,:]==10] = 9
	    top[0].data[0,0,top[0].data[0,0,:,:]==100] = 10

	#plt.subplot(1,2,1)
	#plt.imshow(bottom[0].data[0,0,:,:])
	#plt.subplot(1,2,2)
	#plt.imshow(top[0].data[0,0,:,:])
	#plt.show()

    def backward(self, top, propagate_down, bottom):
	# do nothing
	return
