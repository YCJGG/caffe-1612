import unittest
import tempfile
import os
import six
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import caffe

def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet'
	force_backward: true
        input: 'data'
	input_shape { dim:1 dim: 300 dim: 512 dim: 512}
        input: 'label'
	input_shape { dim:1 dim: 300 dim: 64 dim: 64}
        layer {
	 type: 'Resize'
	 name: 'resize'
	 bottom: 'data'
	 bottom: 'label'
	 top: 'resize'
         resize_param {
	   function_type: BLOB_ALIGN
	   intepolation_type: NEAREST
	   output_size: 128
	 } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
	caffe.set_mode_cpu()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
	img = np.zeros((300,512,512)).astype(float)
	for i in range(100):
		tmp = misc.imread('../examples/images/cat.jpg')
		img[i*3:(i+1)*3,...] = misc.imresize(tmp, [512,512]).transpose(2,0,1)
	self.net.blobs['data'].data[...] = img
	time1 = time.time()
        self.net.forward()
	print '!!!!!\n!!!!!\n!!!!!\n!!!!!\n!!!!!\n!!!!!\n', time.time()-time1
	plt.subplot(1,2,1)
	plt.imshow(self.net.blobs['data'].data[0,0:3,...].transpose(1,2,0))
	plt.subplot(1,2,2)
	plt.imshow(self.net.blobs['resize'].data[0,0:3,...].transpose(1,2,0))
	plt.show()

    def test_backward(self):
	return
