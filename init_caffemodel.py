import numpy as np
#import lmdb
import scipy.io as sio
import scipy.misc as smi
import scipy.ndimage as snd
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import pdb
import sys

import json
from pprint import pprint

sys.path.insert(0, '../../python')
import caffe


def main():
    net_path_train = 'upconv_7.prototxt'

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(net_path_train,caffe.TEST)

    with open(sys.argv[1]) as data_file:    
        data = json.load(data_file)


    net.params['conv1_layer'][0].data[...] = np.array(data[0]['weight'],dtype = np.float32)
    net.params['conv1_layer'][1].data[...] = np.array(data[0]['bias'],dtype = np.float32)

    net.params['conv2_layer'][0].data[...] = np.array(data[1]['weight'],dtype = np.float32)
    net.params['conv2_layer'][1].data[...] = np.array(data[1]['bias'],dtype = np.float32)

    net.params['conv3_layer'][0].data[...] = np.array(data[2]['weight'],dtype = np.float32)
    net.params['conv3_layer'][1].data[...] = np.array(data[2]['bias'],dtype = np.float32)

    net.params['conv4_layer'][0].data[...] = np.array(data[3]['weight'],dtype = np.float32)
    net.params['conv4_layer'][1].data[...] = np.array(data[3]['bias'],dtype = np.float32)

    net.params['conv5_layer'][0].data[...] = np.array(data[4]['weight'],dtype = np.float32)
    net.params['conv5_layer'][1].data[...] = np.array(data[4]['bias'],dtype = np.float32)

    net.params['conv6_layer'][0].data[...] = np.array(data[5]['weight'],dtype = np.float32)
    net.params['conv6_layer'][1].data[...] = np.array(data[5]['bias'],dtype = np.float32)

    net.params['conv7_layer'][0].data[...] = np.array(data[6]['weight'],dtype = np.float32)
    net.params['conv7_layer'][1].data[...] = np.array(data[6]['bias'],dtype = np.float32)


    net.save(sys.argv[2])

if __name__ == "__main__":
    main()


