#
# Provide customized factor (top[0]).
# Author: Wei Zhen @ IIE, CAS
# Finish on: 2017-04-29

import caffe
import numpy as np

class FactorProviderLayer(caffe.Layer):

    def setup(self, bottom, top):
	# check input/output length, one top is accepted, no bottoms
	if len(top) != 1:
	    raise Exception("Factor Provider layer only accepts one top")
	if len(bottom) != 0:
	    raise Exception("Factor Provider layer accepts no bottoms")
	self.iter_counter_ = 0

    def reshape(self, bottom, top):
	top[0].reshape(1, 1, 1, 1)
	## plan1: fix 1.56
	#top[0].data[0,0,0,0] = 1.56
	## plan2: fix 1.35 @ iter 0~2000; fix 1.56 @ 2000~9600
	top[0].data[0,0,0,0] = 1

    def forward(self, bottom, top):
	## plan1: fix 1.56
	#top[0].data[0] = 1.56
	## plan2: fix 1.35 @ iter 0~2000; fix 1.56 @ 2000~9600
	#if self.iter_counter_ < 2000:
	#    top[0].data[0] = 1.35
	#else:
	#    top[0].data[0] = 1.56
	#self.iter_counter_ += 1
	return

    def backward(self, top, propagate_down, bottom):
	return
