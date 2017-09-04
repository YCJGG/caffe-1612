import unittest
import tempfile
import os
import six
import numpy as np

import caffe

def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim:1 dim: 2 dim: 2 dim: 2 }
        input: 'label' input_shape { dim:1 dim: 1 dim: 4 dim: 4 }
        layer { type: 'CenterLoss' name: 'loss' bottom: 'data' bottom: 'label' top: 'loss'
          center_loss_param { label_num: 4 } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
	caffe.set_mode_cpu()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    """def test_forward(self):
        #data = np.array([1,1,2,2,0,0,1,1]).reshape(1,2,2,2)
	#label = np.array([1,2,3,4,4,3,2,1,1,1,2,2,3,3,4,4]).reshape(1,1,4,4)-1
	#loss = np.array([1])
        #self.net.blobs['data'].data[...] = data
	#self.net.blobs['label'].data[...] = label
        self.net.forward()
	#self.assertEqual(self.net.blobs['loss'].data[0], loss)

    def test_backward(self):
	return"""
