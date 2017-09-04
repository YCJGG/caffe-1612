import unittest
import tempfile
import os
import six
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import caffe

in_size = 122
out_size = 61

def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet'
	force_backward: true
        input: 'data'
	input_shape { dim:1 dim: 3 dim: 122 dim: 122}
        input: 'label'
	input_shape { dim:1 dim: 3 dim: 61 dim: 61}
        layer {
	 type: 'Resize'
	 name: 'resize'
	 bottom: 'data'
	 bottom: 'label'
	 top: 'resize'
         resize_param {
	   function_type: BLOB_ALIGN
	   intepolation_type: BILINEAR
	   #output_size: 128
	 } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
	caffe.set_mode_gpu()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
	img = np.zeros((3,in_size,in_size)).astype(float)
	tmp = misc.imread('../examples/images/cat.jpg')
	img[0:3,...] = misc.imresize(tmp, [in_size,in_size]).transpose(2,0,1)
	self.net.blobs['data'].data[...] = img
	time1 = time.time()
        self.net.forward()
	print 'forward !!!!!\n!!!!!\n!!!!!\n!!!!!\n!!!!!\n!!!!!\n', time.time()-time1
	plt.subplot(1,2,1)
	plt.imshow(self.net.blobs['data'].data[0,0:3,...].transpose(1,2,0).astype(np.uint8))
	plt.subplot(1,2,2)
	plt.imshow(self.net.blobs['resize'].data[0,0:3,...].transpose(1,2,0).astype(np.uint8))
	plt.show()

    def test_backward(self):
	img = np.zeros((3,out_size,out_size)).astype(float)
	tmp = misc.imread('../examples/images/cat.jpg')
	img[...] = misc.imresize(tmp, [out_size,out_size]).transpose(2,0,1)
	self.net.blobs['resize'].diff[...] = img
        self.net.backward()
	plt.subplot(1,2,1)
	plt.imshow(self.net.blobs['data'].diff[0,0:3,...].transpose(1,2,0).astype(np.uint8))
	plt.subplot(1,2,2)
	plt.imshow(self.net.blobs['resize'].diff[0,0:3,...].transpose(1,2,0).astype(np.uint8))
	plt.show()
	return
