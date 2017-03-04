import caffe
import numpy as np
import matplotlib.pyplot as plt

class SalientAreaLayer(caffe.Layer):

    def setup(self, bottom, top):
	return

    def reshape(self, bottom, top):
	top[0].reshape(*bottom[0].data.shape)
	top[0].data[...] = np.zeros_like(bottom[0].data)

    def forward(self, bottom, top):
	for batch in range(bottom[0].num):
	    top[0].data[batch,0,(bottom[0].data[batch,0,...]!=0) & (bottom[0].data[batch,0,...]!=255)] = 1
	#plt.subplot(1,3,1)
	#plt.imshow(top[0].data[0,0,:,:])
	#plt.subplot(1,3,2)
	#plt.imshow(bottom[0].data[0,0,:,:])
	#plt.subplot(1,3,3)
	#plt.imshow(bottom[1].data[0,:,:,:].transpose(1,2,0))
	#plt.show()

    def backward(self, top, propagate_down, bottom):
	return

