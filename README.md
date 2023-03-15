# Top-K Confidence Map Aggregation for Robust Semantic Segmentation Against Unexpected Degradation
Yu Moriyasu; Takashi Shibata; Masayuki Tanaka; Masatoshi Okutomi

This is the public code of https://ieeexplore.ieee.org/abstract/document/10043389

## Installation 
```
# Get Top-K Multi-Scale Semantic Sgementation Code
git clone --recursive https://github.com/yumoriyasu/topk-multiscale
cd topk-multiscale

# Get Semantic Segmentation Source Code
git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
```
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


## Demo for [Cityscapes](https://www.cityscapes-dataset.com/)
You need to write the settings in test.sh

image_dir is Cityscapes image datasets directory.
Images' name are "~-leftImg8bit.png"

true_label_dir is Cityscapes true labels directory.
Labels' name are "~-gtFine_labelIds.png"

You can try four types of degradation: JPEG compression, Gaussian noise, Gaussian blur, and N+J compound degradation.
For each, set noise_type to jpeg, noise, blur, or noise+jpeg.

```
# Demo of Semantic Segmentation for Degradation
sh test.sh
```
