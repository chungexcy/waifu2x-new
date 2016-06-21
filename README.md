# Waifu2x-caffe python demo implementation for the upcoming scale-noise fusion model

**Recommand for CNN developer**



This implementation is only a demo of waifu2x. Standard Linux Caffe environment is required. (Ubuntu 14.04/15.04, CUDA 7.5, cuDNN, BLAS, OpenCV, Python, numpy, scipy, pip, pillow, etc.). Online example can be found [here](http://waifu2x-dev.udp.jp/).


### Part I : Introduction


This is a reimplemention from the original [waifu2x](https://github.com/nagadomi/waifu2x/tree/upconv). This demo implemented for the 4 new models only:

    scale2.0x_model
    noise1_scale2.0x_model
    noise2_scale2.0x_model
    noise3_scale2.0x_model

The rest of denoise model is not supported, which can be found in other implementations. Also the scale is fixed for 2x.


Since the implementation is based on python, for demo purpose.The efficience is not good enough for large images processing.


### Part II : Basic Installation

This is a standard caffe application on Linux. The following are required.

    Ubuntu 14.04/15.04
	CUDA 7.5
	cuDNN v5
    Boost
    BLAS via ATLAS
	OpenCV
	Python 2/3, numpy, scipy, pip, pillow

Compiling the Caffe is required first. Code is [here](https://github.com/BVLC/caffe)

Instruction can be found [here](http://caffe.berkeleyvision.org/installation.html) and [here](https://gist.github.com/titipata/f0ef48ad2f0ebc07bcb9)

Note: Matlab is not required. Therefore, noneed for matcaffe compilling.

### Part III : Basic usage of codes

After finished installing, please download the entire folder, and place them into caffe-master/examples/waifu2x-new.

**Running sample of image_test.py**:
```sh
python image_test.py -i test1.png -o test1_waifu2x.png -m scale -cw 350 -ch 300  --model_dir model
```
```sh
python image_test.py -i test2.png -o test2_waifu2x.png -m noise_scale -n 1 -cw 600 -ch 300
```

Help documents
```sh
usage: image_test.py [-h] [-m MODEL] [--model_dir MODEL_DIR] [-n NOISE_LEVEL] [-i INPUT_FILE] [-o OUTPUT_FILE] [-cw CROP_WIDTH] [-ch CROP_HEIGHT] [-c CROP_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        scale|noise_scale
  --model_dir MODEL_DIR
                        Model dir
  -n NOISE_LEVEL, --noise_level NOISE_LEVEL
                        0|1|2|3
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input image address
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output image address
  -cw CROP_WIDTH, --crop_Width CROP_WIDTH
                        Width block, best divided by width
  -ch CROP_HEIGHT, --crop_height CROP_HEIGHT
                        Height block, best divided by height
  -c CROP_SIZE, --crop_size CROP_SIZE
                        Block, best to set ch/cw individually

```

CROP_WIDTH * CROP_HEIGHT are used for splite image into several pieces. You can also assing CROP_SIZE only, where CROP_WIDTH = CROP_HEIGHT = CROP_SIZE.

For best performance, I encourage you to set a relative large value (>200) for CROP_WIDTH and CROP_HEIGHT. And CROP_WIDTH is best to divide width and CROP_HEIGHT is best to divide height. (ex. 300 x 200 for a image wiht 900 x 800 pixels)

-------------------------

**Additional Step**: Update caffe model with new json model

you can update model by yourself using init_caffemodel.py.
```sh
$ python init_caffemodel.py models/noise3_scale2.0x_model.json models/noise3_scale2.0x_model.caffemodel
```

### Part IV : Thanks

Thanks for [waifu2x](https://github.com/nagadomi/waifu2x/tree/upconv) providing the cnn model. Thanks for [caffe](https://github.com/BVLC/caffe) providing the running environment.

