import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from models.vgg_refine import vgg16
from models.se_vgg_refine import se_vgg16
from models.resnet_refine import resnet50, resnet18
from utils.decode import decode_seg_map_sequence
from utils.LoadData_refine import test_data_loader
from utils.Metrics import Cls_Accuracy, RunningConfusionMatrix, IOUMetric
from utils.decode import decode_segmap

parser = argparse.ArgumentParser(description='DRS pytorch implementation')
parser.add_argument("--input_size", type=int, default=320)
parser.add_argument("--crop_size", type=int, default=320)
parser.add_argument("--img_dir", type=str, default="/data/DB/VOC2012/")
parser.add_argument("--train_list", type=str, default='VOC2012_list/train_aug_cls.txt')
parser.add_argument("--test_list", type=str, default='VOC2012_list/train_cls.txt')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

parser.add_argument("--model", type=str, default='vgg16')  # 'vgg16', 'se_vgg16', 'resnet50', 'resnet18'
args = parser.parse_args()
print(args)


""" model load """
if args.model == 'resnet50':
    model = resnet50(num_classes=args.num_classes)
elif args.model == 'resnet18':
    model = resnet18(num_classes=args.num_classes)
elif args.model == 'se_vgg16':
    model = se_vgg16()
else:
    model = vgg16()

model = model.cuda()
model.eval()

ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)


""" dataloader """
test_loader = test_data_loader(args)

""" metric """
mIOU = IOUMetric(num_classes=21)
running_confusion_matrix = RunningConfusionMatrix(labels=range(21))

for idx, dat in enumerate(test_loader):
    print("[%03d/%03d]" % (idx, len(test_loader)), end="\r")

    img, label, sal_map, gt_map, att_map, img_name = dat
    
    label = label.cuda()
    img = img.cuda()

    _, H, W = sal_map.shape
    localization_maps = np.zeros((20, H, W), dtype=np.float32)
        
    # multi-scale testing
    for s in [256, 320, 384]:
        _img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)

        refined_map = model(_img, label, size=(H, W))

        """ obtain refined_map """
        refined_map = refined_map[0].cpu().detach().numpy()
        localization_maps = np.maximum(localization_maps, refined_map)

    img = img[0].cpu().detach().numpy()
    gt_map = gt_map[0].detach().numpy()
    sal_map = sal_map[0].detach().numpy()

    
    """ segmentation label generation """
    localization_maps[localization_maps < args.alpha] = 0 # object cue
    
    bg = np.zeros((1, H, W), dtype=np.float32)
    pred_map = np.concatenate([bg, localization_maps], axis=0)  # [21, H, W]
    
    pred_map[0, :, :] = (1. - sal_map) # backgroudn cue
    
    pred_map = pred_map.argmax(0)
    mIOU.add_batch(pred_map[None, ...], gt_map[None, ...])
    
    
""" performance """
res = mIOU.evaluate()

val_miou = res["Mean_IoU"]
val_pixel_acc = res["Pixel_Accuracy"]

print("\n=======================================")
print("ckpt : ", args.checkpoint)
print("val_miou : %.4f" % val_miou)
print("val_pixel_acc : %.4f" % val_pixel_acc)
print("=======================================\n")
