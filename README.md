# Top-K Confidence Map Aggregation for Robust Semantic Segmentation Against Unexpected Degradation
Yu Moriyasu; Takashi Shibata; Masayuki Tanaka; Masatoshi Okutomi

This is the public code of https://ieeexplore.ieee.org/abstract/document/10043389

## Installation 
This code uses [NVIDIA's semantic segmentation code](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet).
You need to download this code and pre-trained model for Cityscapes.

NVIDIA's code requires:
* An NVIDIA GPU and CUDA 9.0 or higher. Some operations only have gpu implementation.
* PyTorch (>= 0.5.1)
* Python 3
* numpy
* sklearn
* h5py
* scikit-image
* pillow
* piexif
* cffi
* tqdm
* dominate
* tensorboardX
* opencv-python
* nose
* ninja

Our code requires:
* imagedegrade
* openpyxl

[imagedegrade](https://github.com/mastnk/imagedegrade) is a python package to degrade image data.


```
# Get Top-K Multi-Scale Semantic Sgementation Code
git clone --recursive https://github.com/yumoriyasu/topk-multiscale
cd topk-multiscale

# Get Semantic Segmentation Source Code
git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
```


## Demo
You need to write the settings in test.sh
```
# Demo of Semantic Segmentation for Degradation
sh test.sh
```
