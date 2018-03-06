#
# Fusing the two input bottoms according to the cls results given in bottom[2].
#	for soft mode:
#		top = bottom[0]*bottom[2].data[0] + bottom[1]*bottom[2].data[1]
#	for hard mode:
#		top = bottom[2].data[0] > bottom[2].data[1] ? bottom[0]: bottom[1]
# Used in Semantic Preserving Retargeting project
# 
# Author: Wei Zhen @ IIE, CAS & Yingcai, UESTC
# Reimplement on 2018-03-05

import caffe
import numpy as np
import time

class SwitchLayer(caffe.Layer):

    def setup(self, bottom, top):
	# check input/output length, one top and one bottoms are accepted
	if len(top) != 1:
	    raise Exception("SwitchLayer only accepts one top")
	if len(bottom) != 3:
	    raise Exception("SwitchLayer accepts three bottoms")

    def reshape(self, bottom, top):
	top_shape = np.array(bottom[0].data.shape)
	top[0].reshape(*top_shape)

    def forward(self, bottom, top):
	if self.param_str == 'soft':
	    top[0].data[:,:,:,:] = bottom[0].data[:,:,:,:] * bottom[2].data[:,0,:,:] + bottom[1].data[:,:,:,:] * bottom[2].data[:,1,:,:]
	elif self.param_str == 'hard':
	    selection = bottom[2].data.argmax(axis=1).squeeze()
	    for i in range(bottom[2].num()):
		top[0].data[i,:,:,:] = bottom[selection[i]].data[i,:,:,:]
	else:
		raise Exception("Unkonwn fusion mode.")

    def backward(self, top, propagate_down, bottom):
	if self.param_str == 'soft':
	    bottom[0].diff = top[0].diff / (bottom[2].data[:,0,:,:] + 1e30)
	    bottom[1].diff = top[0].diff / (bottom[2].data[:,1,:,:] + 1e30)
	elif self.param_str == 'hard':
	    selection = bottom[2].data.argmax(axis=1).squeeze()
	    for i in range(bottom[2].num()):
		bottom[selection[i]].diff[i,:,:,:] = top[0].diff[i,:,:,:]
	else:
		raise Exception("Unkonwn fusion mode.")
