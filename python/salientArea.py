import caffe
import numpy as np
import matplotlib.pyplot as plt

class SalientAreaLayer(caffe.Layer):

    def setup(self, bottom, top):
	return

    def reshape(self, bottom, top):
	top_shape = np.array(bottom[0].data.shape)
	if bottom[0].channels > 1:
	    top_shape[1] = 1
	    self.bottom_data = bottom[0].data.argmax(axis=1).reshape(top_shape)
	else:
	    self.bottom_data = bottom[0].data
	top[0].reshape(*top_shape)
	top[0].data[...] = np.zeros_like(self.bottom_data)

    def forward(self, bottom, top):
	for batch in range(bottom[0].num):
	    top[0].data[batch,0,(self.bottom_data[batch,0,...]!=0) & (self.bottom_data[batch,0,...]!=255)] = 1
	#plt.subplot(1,3,1)
	#plt.imshow(top[0].data[0,0,:,:])
	#plt.subplot(1,3,2)
	#plt.imshow(self.bottom_data[0,0,:,:])
	#plt.subplot(1,3,3)
	#plt.imshow(bottom[1].data[0,:,:,:].transpose(1,2,0))
	#plt.show()

    def backward(self, top, propagate_down, bottom):
	return

