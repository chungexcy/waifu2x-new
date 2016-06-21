import numpy as np
import scipy.misc
from PIL import Image, ImageOps
import scipy.io
import os
import time
import datetime
import sys
import argparse

sys.path.insert(0, '../../python')
import caffe

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="scale|noise_scale")
    parser.add_argument("--model_dir", help="Model dir")
    parser.add_argument("-n", "--noise_level", help="0|1|2|3")
    parser.add_argument("-i", "--input_file", help="Input image address")
    parser.add_argument("-o", "--output_file", help="Output image address")
    parser.add_argument("-cw", "--crop_Width", help="Width block, best divided by width")
    parser.add_argument("-ch", "--crop_height", help="Height block, best divided by height")
    parser.add_argument("-c", "--crop_size", help="Block, best to set ch/cw individually")
    
    args = parser.parse_args()
    
    if args.model_dir:
        net = args.model_dir
        net.rstrip('/')
        net += "/"
    else:
        net = "models/"
    
    if args.model:
        if args.model == "scale":
            net += "scale2.0x_model.caffemodel"
        elif args.model == "noise_scale":
            if args.noise_level == "0":
                net += "scale2.0x_model.caffemodel"
            elif args.noise_level == "1":
                net += "noise1_scale2.0x_model.caffemodel"
            elif args.noise_level == "2":
                net += "noise2_scale2.0x_model.caffemodel"
            elif args.noise_level == "3":
                net += "noise3_scale2.0x_model.caffemodel"
            else:
                exit()
        else:
            exit()
    input_file = args.input_file
    output_file = args.output_file
    
    if args.crop_size:
        block_Width = int(args.crop_size)
        block_height = int(args.crop_size)
    else:
        block_Width = 128
        block_height = 128
    
    if args.crop_Width:
        block_Width = int(args.crop_Width)
    if args.crop_height:
        block_height = int(args.crop_height)
        
    return [input_file, output_file, net, block_Width, block_height]
    

def getblock(image, width_block, height_block, border):
    width = len(image[0][0])
    height = len(image[0])
    if width_block > width - 2*border:
        width_block = width - 2*border
    if height_block > height - 2*border:
        height_block = height - 2*border
    
    target_width = (width - 2*border)*2
    target_height = (height - 2*border)*2
    
    width_num = (width - 2*border)/width_block
    if (width - 2*border) > width_num * width_block:
        width_num += 1
    height_num = (height - 2*border)/height_block
    if (height - 2*border) > height_num * height_block:
        height_num += 1
        
    output = np.zeros((3, target_height, target_width), dtype=np.float)
    
    patch = list()
    region = list()
    
    for i in range(width_num):
        for j in range(height_num):
            x1 = i*width_block;
            x2 = min(x1+width_block+2*border, width)
            y1 = j*height_block;
            y2 = min(y1+height_block+2*border, height)
            patch.append(image[:,y1:y2,x1:x2])
            
            x1_ =2*i*width_block
            x2_ = min(x1_+2*width_block, target_width)
            y1_ = 2*j*height_block;
            y2_ = min(y1_+2*height_block, target_height)
            region.append([y1_,y2_,x1_,x2_])
    
    return [patch, region, output]


def main():
    [input_file, output_file, net, block_Width, block_height] = arg()
    #exit()
    #remove the following two lines if testing with cpu
    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net('upconv_7.prototxt', net, caffe.TEST)

    im = Image.open(input_file)
    in2_ = np.array(im, dtype=np.float32)
    #im = im.resize((200,139), Image.BICUBIC)
    im = ImageOps.expand(im, border=7,fill='black')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_ / 255.0
    in_ = in_.transpose((2,0,1))

    time1 = datetime.datetime.now()

    [patch, region, output] = getblock(in_, block_Width, block_height, 7)

    for idx in range(len(patch)):
        
        net.blobs['input'].reshape(1, *(patch[idx]).shape)
        net.blobs['input'].data[...] = (patch[idx])
        
        net.forward()
        output_idx = net.blobs['conv7'].data[0][:,:,:]
        
        output[:,region[idx][0]:region[idx][1],region[idx][2]:region[idx][3]] = output_idx

    output = np.minimum(output,1.0)
    output = np.maximum(output,0.0)
    output = np.rint(output*255.0)
    output = np.array(output, dtype=np.uint8)
        
    scipy.misc.imsave(output_file, output)


    time2 = datetime.datetime.now() # waited a few minutes before pressing enter
    elapsedTime = time2 - time1
    print (elapsedTime)

if __name__ == "__main__":
    main()
            
