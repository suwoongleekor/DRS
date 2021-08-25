import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import time

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from models.vgg import vgg16
from models.vgg_CAAS import vgg16
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

import pudb


def get_arguments():
    parser = argparse.ArgumentParser(description='DRS pytorch implementation')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='VOC2012_list/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='VOC2012_list/train_cls.txt')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument('--logdir', default='logs/test', type=str, help='Tensorboard log dir')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0, help='set 0 for the learnable DRS')
    parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

    parser.add_argument("--loss_cls_aware_mode", type=str, default='None')  # 'None', 'cls_aware'
    parser.add_argument("--lambda_cls_aware", type=float, default=0.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--from_scratch", action="store_true")

    return parser.parse_args()


def get_model(args):
    if args.from_scratch:
        model = vgg16(pretrained=False, delta=args.delta)
    else:
        model = vgg16(pretrained=True, delta=args.delta)

    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], 
        momentum=0.9, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )

    return  model, optimizer


def validate(current_epoch):
    print('\nvalidating ... ', flush=True, end='')
    
    mIOU = IOUMetric(num_classes=21)
    cls_acc_matrix = Cls_Accuracy()
    
    val_loss = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img, label, sal_map, gt_map, _ = dat
            
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit, cam = model(img, label)

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            val_loss.update(loss.data.item(), img.size()[0])
            
            """ obtain CAMs """
            cam = cam.cpu().detach().numpy()
            gt_map = gt_map.detach().numpy()
            sal_map = sal_map.detach().numpy()

            """ segmentation label generation """
            cam[cam < args.alpha] = 0  # object cue
            bg = np.zeros((B, 1, H, W), dtype=np.float32)
            pred_map = np.concatenate([bg, cam], axis=1)  # [B, 21, H, W]

            pred_map[:, 0, :, :] = (1. - sal_map) # background cue
            pred_map = pred_map.argmax(1)

            mIOU.add_batch(pred_map, gt_map)
    
    """ validation score """
    res = mIOU.evaluate()
    val_miou = res["Mean_IoU"]
    val_pixel_acc = res["Pixel_Accuracy"]
    val_cls_acc = cls_acc_matrix.compute_avg_acc()
    
    """ tensorboard visualization """
    for n in range(min(4, img.shape[0])):
        result_vis  = output_visualize(img[n], cam[n], label[n], gt_map[n], pred_map[n])
        writer.add_images('valid output %d' % (n+1), result_vis, current_epoch)
        
    writer.add_scalar('valid loss', val_loss.avg, current_epoch)
    writer.add_scalar('valid acc', val_cls_acc, current_epoch)
    writer.add_scalar('valid mIoU', val_miou, current_epoch)
    writer.add_scalar('valid Pixel Acc', val_pixel_acc, current_epoch)
    
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    
    return val_miou
    

def train(current_epoch):
    train_loss = AverageMeter()
    train_loss_cls = AverageMeter()
    train_loss_cls_aware = AverageMeter()

    cls_acc_matrix = Cls_Accuracy()
    cls_aware_acc_matrix = Cls_Accuracy()

    model.train()

    cls_aware_scale = np.array([0, 0, 1./14, 1./14, 1./14, 1./14, 1./14, 1./14, 1./14, 1./14
                                , 1./14, 1./14, 1./14, 1./14, 1./14, 1./14])
    print('cls_aware_scale: ', cls_aware_scale)

    cls_aware_scale_pt = torch.from_numpy(cls_aware_scale)
    cls_aware_scale_pt = cls_aware_scale_pt.to('cuda')

    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):
        # if idx>50:
        #     break

        img, label, _ = dat
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit, route_outs, route_classifier_outs = model(img)

        loss_cls_aware = torch.zeros([1], device='cuda')

        if args.loss_cls_aware_mode == 'cls_aware':
            for j, route_classifier_out in enumerate(route_classifier_outs):
                # print('route_classifier_out: ', route_classifier_out)
                # print('route_classifier_out.shape: ', route_classifier_out.shape)
                # print('cls_aware_scale_pt.shape: ', cls_aware_scale_pt.shape)
                # print('cls_aware_scale_pt[j]: ', cls_aware_scale_pt[j])
                loss_cls_aware += (F.multilabel_soft_margin_loss(route_classifier_out, label) * cls_aware_scale_pt[j])
        elif args.loss_cls_aware_mode == 'None':
            pass
        else:
            print('Not supported loss_att_mode')



        """ classification loss """
        loss_cls = F.multilabel_soft_margin_loss(logit, label)

        loss = args.lambda_cls * loss_cls + args.lambda_cls_aware * loss_cls_aware

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_acc_matrix.update(logit, label)
        cls_aware_acc_matrix.update(route_classifier_outs[-1], label)

        train_loss.update(loss.data.item(), img.size()[0])
        train_loss_cls.update(loss_cls.data.item(), img.size()[0])
        train_loss_cls_aware.update(loss_cls_aware.data.item(), img.size()[0])
        
        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            train_cls_acc = cls_acc_matrix.compute_avg_acc()
            train_cls_aware_acc = cls_aware_acc_matrix.compute_avg_acc()

            writer.add_scalar('train loss', train_loss.avg, global_counter)
            writer.add_scalar('train loss cls', train_loss_cls.avg, global_counter)
            writer.add_scalar('train loss cls aware', train_loss_cls_aware.avg, global_counter)

            writer.add_scalar('train acc', train_cls_acc, global_counter)
            writer.add_scalar('train acc cls aware', train_cls_aware_acc, global_counter)

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'ACC: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, idx+1, len(train_loader),
                    optimizer.param_groups[0]['lr'], 
                    train_cls_acc, loss=train_loss)
                 )

    args.global_counter = global_counter
    
    
if __name__ == '__main__':
    # pu.db

    args = get_arguments()
    
    nGPU = torch.cuda.device_count()
    print("start training the classifier with DRS , nGPU = %d" % nGPU)
    
    args.batch_size *= nGPU
    args.num_workers *= nGPU
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    best_score = 0
    model, optimizer = get_model(args)

    for current_epoch in range(1, args.epoch+1):
        
        train(current_epoch)
        score = validate(current_epoch)
        
        """ save checkpoint """
        if score > best_score:
            best_score = score
            print('\nSaving state, epoch : %d , mIoU : %.4f \n' % (current_epoch, score))
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter,
                'miou': score,
            }

            # pu.db

            state_dict = state['model']
            key_to_delete = []
            for k in state_dict.keys():
                if 'classifier_route' in k:
                    key_to_delete.append(k)

            for k in key_to_delete:
                del state_dict[k]

            model_file = os.path.join(args.save_folder, 'best.pth')
            torch.save(state, model_file)
