import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './semantic-segmentation'))
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

import imagedegrade.np as degrade
from skimage.transform import resize
import openpyxl as excel


def calculate_miou(pred_image, true_label):
    label_id = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

    shape = true_label.shape
    remove_area = np.zeros(shape, dtype="int")
    for i in range(19):
        true_label01 = np.where(true_label == label_id[i], 1, 0)
        remove_area = remove_area | true_label01
        remove_area = remove_area.astype(np.uint8)
    pred_image_removed = pred_image * remove_area
    pred_image_removed = pred_image_removed.astype(np.uint8)

    TP = np.zeros(label_id.size)
    FP = np.zeros(label_id.size)
    FN = np.zeros(label_id.size)

    for i in range(label_id.size):
        tru_im01 = np.where(true_label == label_id[i], 1, 0)
        pre_im01 = np.where(pred_image_removed == label_id[i], 1, 0)

        intersection = np.sum(tru_im01 * pre_im01)
        TP[i] += intersection
        FP[i] += np.sum(pre_im01) - intersection
        FN[i] += np.sum(tru_im01) - intersection

    return TP, FP, FN


parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--image_dir', type=str, default='', help='path to the folder containing demo images', required=True)
parser.add_argument('--true_label_dir', type=str, default='', help='path to the folder containing true labels', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--noise_type', type=str, help='Degrade Type')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

# get data
data_dir = args.image_dir
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

# added component
true_dir = args.true_label_dir
true_list = os.listdir(true_dir)
true_list.sort()
noise_type = str(args.noise_type)
if noise_type == 'jpeg':
    quality_range = list(range(20, 81, 20))
elif noise_type == 'noise':
    quality_range = list(range(4,21,4))
elif noise_type == 'blur':
    quality_range = list(range(1,6,1))
elif noise_type == 'noise+jpeg':
    noise_sigma = list(range(5,11,5))
    jpeg_quality = list(range(40,81,40))
    quality_range = list( range( len(noise_sigma)*len(jpeg_quality) ) )
num_quality = len(quality_range)

resize_rate = 0.8
resize_para_list = np.power(resize_rate, range(5))
num_scale = len(resize_para_list)

mIoU = np.zeros([19])
pred_ensemble = np.zeros( [ 1024, 2048, 19, num_scale ] )

for qidx in range( num_quality ):
    TP = np.zeros([19])
    FP = np.zeros([19])
    FN = np.zeros([19])
    img_id = 0

    if noise_type == 'noise+jpeg':
        title = "ensemble_noise"+str(noise_sigma[int(quality_range[qidx] // len(jpeg_quality))])+"_jpeg"+str(jpeg_quality[int(quality_range[qidx] % len(jpeg_quality))])+".xlsx"
        title = "./result/" + title
    else:
        title = "ensemble_"+str(noise_type)+str(quality_range[qidx])+".xlsx"
        title = "./result/" + title
    
    wb = excel.Workbook()
    ws = wb.active

    for img_name, true_name in zip(images, true_list):
        img_id += 1
        true_label_dir = os.path.join(true_dir, true_name)
        true_label = np.array(Image.open(true_label_dir).convert("L"), dtype="int")

        img_dir = os.path.join(data_dir, img_name)
        img_input = np.array(Image.open(img_dir).convert('RGB'))

        img_width = img_input.shape[1]
        img_height = img_input.shape[0]

        # Create Degraded Image
        if noise_type == 'jpeg':
            img_degrade = degrade.jpeg(img_input, jpeg_quality=quality_range[qidx])
        elif noise_type == 'noise':
            img_degrade = degrade.noise(img_input, noise_sigma=quality_range[qidx])
        elif noise_type == 'blur':
            img_degrade = degrade.blur(img_input, blur_sigma=quality_range[qidx])
        elif noise_type == 'noise+jpeg':
            img_degrade = degrade.noise( img_input, noise_sigma=noise_sigma[int(quality_range[qidx] // len(jpeg_quality))] )
            img_degrade = np.clip(img_degrade, 0, 255)
            img_degrade = img_degrade.astype("uint8")
            img_degrade = degrade.jpeg( img_degrade, jpeg_quality=jpeg_quality[int(quality_range[qidx] % len(jpeg_quality))] )
        
        # Save Degraded Image
#        img_save = np.clip(img_degrade, 0, 255)
#        img_save = img_save.astype("uint8")
#        Image.fromarray(img_save).save('./result/degimage_'+img_name)

        # Scales
        for sidx in range( num_scale ):
            img_resize = img_degrade / 255
            img_resize = resize(img_resize, (int(img_height*resize_para_list[sidx]),int(img_width*resize_para_list[sidx])), order=1)
            img_resize = img_resize.astype('float32')
            img_tensor = img_transform(img_resize)

            # predict
            with torch.no_grad():
                pred = net(img_tensor.unsqueeze(0).cuda())
                print('%04d/%04d: Inference done.' % (img_id, len(images)), "Scale is",resize_para_list[sidx])

            pred = pred.cpu().numpy().squeeze()
            pred_trans = pred.transpose(1,2,0)
            pred_ensemble[:,:,:,sidx] = resize(pred_trans, (img_height,img_width))
            
        # top_K
        top_K = 3
        result = np.mean(np.sort(pred_ensemble, axis=3)[:,:,:,(num_scale - top_K):], axis=3) 
        result = np.argmax(result, axis=2)

        # Save Segmentation Result
#        colorized = args.dataset_cls.colorize_mask(result)
#        colorized.save(os.path.join('./result/'+str(noise_type)+str(quality_range[qidx])+'_'+img_name))

        label_out = np.zeros_like(result)
        for label_id, train_id in args.dataset_cls.id_to_trainid.items():
            label_out[np.where(result == train_id)] = label_id

        TP_add, FP_add, FN_add = calculate_miou(label_out, true_label)
        TP += TP_add
        FP += FP_add
        FN += FN_add

    mIoU = TP / (TP + FP + FN)
    ws.cell(1,1).value = np.sum(mIoU)/19

    wb.save(title)
    wb.close()
