#!/usr/bin/env bash

# image_dir is Cityscapes Image Datasets Directory
# Images' name "~~-leftImg8bit.png"

# true_label_dir is Cityscapes True Labels Directory
# Labels' name "~~-gtFine_labelIds.png"

python test.py --image_dir ./Cityscapes/images \
    --true_label_dir ./Cityscapes/true_label \
    --snapshot ./pretrained_models/cityscapes_best.pth \
    --noise_type noise+jpeg
