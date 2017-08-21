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
	input_shape { dim:1 dim: 4 dim: 6 dim: 6}
        layer {
	 type: 'DUCAggregation'
	 name: 'duc'
	 bottom: 'data'
	 top: 'data_up'
         duc_param {
	   upsampling_factor: 2
	 } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
	#caffe.set_mode_gpu()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
	data = np.array(range(144))
	data_up = data.reshape(1,1,12,12)
	data = data.reshape(1,4,6,6)
	self.net.blobs['data'].data[...] = data
#	time1 = time.time()
        self.net.forward()
	print data
	print self.net.blobs['data_up'].data[...]
#	print time.time()-time1
	for i in range(4):
		self.assertEqual(self.net.blobs['data_up'].data.shape[i], data_up.shape[i])
	#for idx, i in enumerate(self.net.blobs['data_up'].data[...]):
	#	self.assertEqual(data_up.flat[idx],i)

    def test_backward(self):
	data = np.array(range(144))
	data_up = data.reshape(1,1,12,12)
	data = data.reshape(1,4,6,6)
	self.net.blobs['data_up'].diff[...] = data_up
        self.net.backward()
	print data_up
	print self.net.blobs['data'].diff[...]
	for i in range(4):
		self.assertEqual(self.net.blobs['data'].data.shape[i], data.shape[i])
	#for idx, i in enumerate(self.net.blobs['data'].data[...]):
	#	self.assertEqual(data.flat[idx],i)
