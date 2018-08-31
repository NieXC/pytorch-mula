import argparse
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from nets.vgg_based_network import MuLA_VGG_MSRAInit
from nets.hourglass_based_network import MuLA_HG_MSRAInit

from utils.data_loader import LIPDataset
from utils.calc_pckh import calc_pck_lip_dataset 
from utils.calc_miou import calc_miou_lip_dataset
import utils.eval_util as eval_util

parser = argparse.ArgumentParser(description='PyTorch Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation on LIP dataset')
parser.add_argument('--train-data', default='dataset/lip/train_images/', metavar='DIR', help='path to training dataset')
parser.add_argument('--train-pose-anno', default='dataset/lip/jsons/LIP_SP_TRAIN_annotations.json', type=str, metavar='PATH', help='path to training pose annotations')
parser.add_argument('--train-parsing-anno', default='dataset/lip/train_segmentations', metavar='DIR', help='path to training parsing annotations')
parser.add_argument('--eval-data', default='dataset/lip/val_images', metavar='DIR', help='path to eval dataset')
parser.add_argument('--eval-pose-anno', default='dataset/lip/jsons/LIP_SP_VAL_annotations.json', type=str, metavar='PATH', help='path to eval pose annotations')
parser.add_argument('--eval-parsing-anno', default='dataset/lip/val_segmentations', metavar='DIR', help='path to eval parsing annotations')

parser.add_argument('--arch', default='HG', type=str, metavar='PATH', help='Network architecture (VGG or HG (Hourglass), default: HG)')

parser.add_argument('-b', '--batch_size', default=10, type=int, metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run (default: 250)')
parser.add_argument('--snapshot-fname-prefix', default='exps/snapshots/mula_lip', type=str, metavar='PATH', help='path to snapshot')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--calc-pck', default=False, type=bool, metavar='BOOL', help='caculate PCK or not')
parser.add_argument('--calc-miou', default=False, type=bool, metavar='BOOL', help='caculate mIoU or not')
parser.add_argument('--pose-pred-path', default='exps/preds/pose_results/pred_keypoints_lip.csv', type=str, metavar='PATH', help='path to save the pose prediction results in .csv format')
parser.add_argument('--parsing-pred-dir', default='exps/preds/parsing_results', metavar='DIR', help='path to save the parsing prediction results in a set of label maps in .png format ')
parser.add_argument('--visualization', default=False, type=bool, metavar='BOOL', help='visualizae prediction or not')
parser.add_argument('--vis-dir', default='exps/preds/vis_results', metavar='DIR', help='path to save visualization results')

best_pck = 0
pck_avg_list = []
pck_all_list = []

best_miou = 0
miou_all_list = []

def main():
    # Global variables
    global args, best_pck, pck_avg_list, pck_all_list, best_miou, miou_all_list
    args = parser.parse_args()

    # Welcome msg
    phase_str = '[Train and Val Phase]'
    if args.evaluate:
        phase_str = '[Testing Phase]'
    print('Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation: {0}'.format(phase_str))

    # Create network
    if args.arch == 'VGG':
        mula_net = MuLA_VGG_MSRAInit()
        pose_net_stride = 8
    elif args.arch == 'HG':
        mula_net = MuLA_HG_MSRAInit()
        pose_net_stride = 4
    else:
        raise RuntimeError('Unknown network architecture!')
        
    # Multi-GPU setting
    mula_net = nn.DataParallel(mula_net).cuda()

    # CUDNN setting
    cudnn.benchmark = True
    cudnn.enabled = True

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> Loading checkpoints {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            mula_net.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_pck = checkpoint['best_pck']
            pck_avg_list = checkpoint['pck_avg_list']
            pck_all_list = checkpoint['pck_all_list']
            best_miou = checkpoint['best_miou']
            miou_all_list = checkpoint['miou_all_list']
            mula_net_params = mula_net.parameters()
        else:
            print('=> No checkpoint found at {0}'.format(args.resume))

    mula_net_params = mula_net.parameters()

    # Snapshot file names
    snapshot_fname = '{0}.pth.tar'.format(args.snapshot_fname_prefix)
    snapshot_pose_best_fname = '{0}_pose_best.pth.tar'.format(args.snapshot_fname_prefix)
    snapshot_parsing_best_fname = '{0}_parsing_best.pth.tar'.format(args.snapshot_fname_prefix)

    # Image normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

    # Data transform
    data_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    # LIP dataset
    lip_ds = LIPDataset(args.train_data, \
                        args.train_pose_anno, \
                        args.train_parsing_anno, \
                        transform=data_transform, \
                        pose_net_stride=pose_net_stride, \
                        parsing_net_stride=1, \
                        crop_size=256, \
                        target_dist=1.171, scale_min=0.8, scale_max=1.5, \
                        max_rotate_degree=40, \
                        max_center_trans=40, \
                        flip_prob=0.5, \
                        is_visualization=False)

    # Load training data 
    train_loader = torch.utils.data.DataLoader(lip_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Load validation data
    print('Loading evaluation json file: {0}...'.format(args.eval_pose_anno))
    eval_list = []
    with open(args.eval_pose_anno) as data_file:
        data_this = json.load(data_file)
        data_this = data_this['root']
        eval_list = eval_list + data_this
    eval_im_name_list = []
    for ii in range(0, len(eval_list)):
        eval_item = eval_list[ii]
        eval_im_name_list.append(eval_item['im_name'])
    print('Finished loading evaluation json file')

    # MSE Loss function for pose estimation and CrossEntropy Loss function for parsing estimation
    pose_criterion = nn.MSELoss().cuda()
    parsing_criterion = nn.NLLLoss2d().cuda()

    # RMSProp as the optimizer
    optimizer = torch.optim.RMSprop(mula_net_params, args.lr)
	
    # Testing 
    if args.evaluate == True:
        evaluate(mula_net, \
                 args.eval_data, \
                 eval_im_name_list, \
                 transform=data_transform, \
                 stride=pose_net_stride, \
                 crop_size=256, \
                 scale_multiplier=[1], \
                 visualization=args.visualization, \
                 vis_result_dir=args.vis_dir, \
                 pose_pred_path=args.pose_pred_path, \
                 parsing_pred_dir=args.parsing_pred_dir, \
                 is_calc_pck=args.calc_pck, \
                 is_calc_miou=args.calc_miou, \
                 eval_num=-1)

        return

    for epoch in range(args.start_epoch, args.epochs):

        # Training
        train(train_loader, mula_net, pose_criterion, parsing_criterion, optimizer, epoch)

        # Save snapshot
        torch.save({ 
            'epoch': epoch + 1,
            'state_dict': mula_net.state_dict(),
            'best_pck': best_pck,
            'pck_avg_list': pck_avg_list,
            'pck_all_list': pck_all_list,
            'best_miou': best_miou,
            'miou_all_list': miou_all_list,
        }, snapshot_fname)

        # Validation 
        if epoch < 150:
            val_freq = 10
        elif epoch < 170:
            val_freq = 5
        elif epoch < 200:
            val_freq = 2
        else:
            val_freq = 1		

        if (epoch + 1) % val_freq == 0:
            pck_avg, miou = evaluate(mula_net, \
                                     args.eval_data, \
                                     eval_im_name_list, \
                                     transform=data_transform, \
                                     stride=pose_net_stride, \
                                     crop_size=256, \
                                     scale_multiplier=[1], \
                                     visualization=args.visualization, \
                                     vis_result_dir=args.vis_dir, \
                                     pose_pred_path=args.pose_pred_path, \
                                     parsing_pred_dir=args.parsing_pred_dir, \
                                     is_calc_pck=True, \
                                     is_calc_miou=True, \
                                     eval_num=2000)
            
            is_pose_best = pck_avg > best_pck
            best_pck = max(pck_avg, best_pck)
            
            is_parsing_best = miou > best_miou
            best_miou = max(miou, best_miou)

            torch.save({ 
                'epoch': epoch + 1,
                'state_dict': mula_net.state_dict(),
                'best_pck': best_pck,
                'pck_avg_list': pck_avg_list,
                'pck_all_list': pck_all_list,
                'best_miou': best_miou,
                'miou_all_list': miou_all_list,
            }, snapshot_fname)
            if is_pose_best:
                shutil.copyfile(snapshot_fname, snapshot_pose_best_fname)
            if is_parsing_best:
                shutil.copyfile(snapshot_fname, snapshot_parsing_best_fname)
		
def train(train_loader, model, pose_criterion, parsing_criterion, optimizer, epoch):

    cur_lr = adjust_learning_rate(optimizer, epoch)

    losses = AverageMeter()
    cost_time = AverageMeter()
    train_pose_acc = AverageMeter()

    model.train()

    iter_start_time = time.time()
    for i, (im, pose_target, parsing_target) in enumerate(train_loader):

        # Prepare input and target variables
        im = im.cuda(async=True)
        pose_target = pose_target.float().cuda(async=True)
        parsing_target = parsing_target.long().cuda(async=True)
        input_var = torch.autograd.Variable(im)
        pose_target_var = torch.autograd.Variable(pose_target)
        parsing_target_var = torch.autograd.Variable(parsing_target)

        # Network forward
        pose_output, parsing_output = model(input_var)

        # Calculate loss
        # Case 1: output are lists from Hourglass network
        # Case 2: output are tensors from VGG network
        total_loss = 0.0
        if isinstance(pose_output, list):
            pose_avg_acc = cal_train_acc(pose_output[-1].data, pose_target)
            for s in range(0, len(pose_output)):
                pose_loss = pose_criterion(pose_output[s], pose_target_var)
                parsing_loss = parsing_criterion(parsing_output[s], parsing_target_var)
                total_loss += (pose_loss + 0.01 * parsing_loss)
        else:
            pose_avg_acc = cal_train_acc(pose_output.data, pose_target)
            pose_loss = pose_criterion(pose_output, pose_target_var)
            parsing_loss = parsing_criterion(parsing_output, parsing_target_var)
            total_loss += (pose_loss + 0.01 * parsing_loss)

        train_pose_acc.update(pose_avg_acc, 1)
        losses.update(total_loss.data[0], im.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        cost_time.update(time.time() - iter_start_time)
        iter_start_time = time.time()
        
        if i == 0 or (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] \t'
                  'CurLR: {3} \t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f}) \t'
                  'Acc {accuracy.val:.3f} ({accuracy.avg:.3f}) \t'
                  'BatchTime {cost_time.val:.3f} ({cost_time.avg:.3f}) \t'.format(
                  epoch + 1, i + 1, len(train_loader), 
                  cur_lr, 
                  loss=losses, 
                  accuracy=train_pose_acc, 
                  cost_time=cost_time))

def evaluate(model, \
             eval_im_root_dir, \
             eval_im_name_list, \
             transform=None, \
             stride=4, \
             crop_size=256, \
             scale_multiplier=[1], \
             num_of_joints=16, \
             num_of_parts=20, \
             visualization=False, \
             vis_result_dir='exps/preds/vis_results', \
             pose_gt_path='dataset/lip/val_gt/lip_val_groundtruth.csv', \
             pose_pred_path='exps/preds/pose_results/pred_keypoints_lip.csv', \
             parsing_gt_dir='dataset/lip/val_segmentations', \
             parsing_pred_dir='exps/preds/parsing_results', \
             is_calc_pck=True, \
             is_calc_miou=True, \
             eval_num=-1):

    model.eval()

    if eval_num <= 0:
        eval_num = len(eval_im_name_list)

    # Evaluate human pose 
    pose_list = eval_util.multi_image_testing_for_pose_on_lip_dataset(model, \
                                                                      eval_im_root_dir, \
                                                                      eval_im_name_list, \
                                                                      transform=transform, \
                                                                      stride=stride, \
                                                                      crop_size=crop_size, \
                                                                      scale_multiplier=scale_multiplier, \
                                                                      num_of_joints=num_of_joints,  \
                                                                      visualization=visualization, \
                                                                      vis_result_dir=vis_result_dir, \
                                                                      eval_num=eval_num)
    eval_util.save_hpe_results_to_lip_format(eval_im_name_list, pose_list, save_path=pose_pred_path, eval_num=eval_num)
    pck_avg = 0.0
    if is_calc_pck:
        pck_all = calc_pck_lip_dataset(pose_gt_path, pose_pred_path, method_name='MuLA', eval_num=eval_num)
        pck_avg = pck_all[-1][-1]
        pck_all_list.append(pck_all)
        pck_avg_list.append(pck_avg)   

    # Evaluate human parsing
    eval_util.multi_image_testing_for_parsing_on_lip_dataset(model, \
                                                             eval_im_root_dir, \
                                                             eval_im_name_list, \
                                                             parsing_pred_dir=parsing_pred_dir, \
                                                             transform=transform, \
                                                             crop_size=crop_size, \
                                                             num_of_parts=num_of_parts, \
                                                             visualization=visualization, \
                                                             vis_result_dir=vis_result_dir, \
                                                             eval_num=eval_num)
    miou = 0.0
    if is_calc_miou:
        miou_all = calc_miou_lip_dataset(eval_im_name_list, parsing_gt_dir, parsing_pred_dir, eval_num=eval_num)
        miou = miou_all['mean_iou']
        miou_all_list.append(miou_all)

    return pck_avg, miou

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    decay = 0
    if epoch + 1 >= 230:
        decay = 0.05  
    elif epoch + 1 >= 200:
        decay = 0.1
    elif epoch + 1 >= 170:
        decay = 0.25
    elif epoch + 1 >= 150:
        decay = 0.5
    else:
        decay = 1

    lr = args.lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Get predictions
def get_preds(heatmaps):
    if heatmaps.dim() != 4:
        raise ValueError('Input must be 4-D tensor')
    max_val, max_idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), heatmaps.size(2) * heatmaps.size(3)), 2)
    preds = torch.Tensor(max_idx.size(0), max_idx.size(1), 2)
    preds[:, :, 0] = max_idx[:, :] % heatmaps.size(3)
    preds[:, :, 1] = max_idx[:, :] / heatmaps.size(3)
    preds[:, :, 1] = preds[:, :, 1].floor()
    return preds

def calc_dists(preds, labels, normalize):
    dists = torch.Tensor(preds.size(1), preds.size(0))
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            if labels[i, j, 0] == 0 and labels[i, j, 1] == 0:
                dists[j, i] = -1
            else:
                dists[j, i] = torch.dist(labels[i, j, :], preds[i, j, :]) / normalize
    return dists

def dist_accuracy(dists, th=0.5):
    if torch.ne(dists, -1).sum() > 0:
        return (dists.le(th).eq(dists.ne(-1)).sum()) * 1.0 / dists.ne(-1).sum()
    else:
        return -1

def cal_train_acc(output, target):

    num_of_joints = target.size(1) - 1 

    preds = get_preds(output)
    gt = get_preds(target)
    dists = calc_dists(preds, gt, output.size(3) / 10.0)

    avg_acc = 0.0
    bad_idx_count = 0
    for ji in range(num_of_joints):
        acc = dist_accuracy(dists[ji, :])
        if acc > 0:
            avg_acc += acc
        else:
            bad_idx_count += 1
    if bad_idx_count != num_of_joints:
        avg_acc = avg_acc / (num_of_joints - bad_idx_count)
    return avg_acc

if __name__ == '__main__':
    main()


