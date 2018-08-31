import argparse
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
import json
import cv2
from scipy.ndimage.filters import gaussian_filter
import csv
import math
from numpy.core.records import fromarrays
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils.data_augmentation import augmentation_cropped
from utils.vis_utils import vis_hpe_results, vis_parsing_results

# Multi-scale Multi-stage testing for Human Pose Estimation on LIP dataset
def multi_image_testing_for_pose_on_lip_dataset(net, \
                                                im_root_dir, \
                                                im_name_list, \
                                                transform=None, \
                                                stride=4, \
                                                crop_size=256, \
                                                scale_multiplier=[1], \
                                                num_of_joints=16, \
                                                visualization=False, \
                                                vis_result_dir='exps/preds/vis_results', \
                                                eval_num=-1):

    print('LIP Testing for pose with flipping multi-scale: {0} scales'.format(len(scale_multiplier)))

    if eval_num > 0:
        num_of_im = eval_num
    else:
        num_of_im = len(im_name_list)

    total_time = 0

    pose_list = np.zeros((num_of_im, num_of_joints, 3))
    for ii in range(0, num_of_im):

        im_name = im_name_list[ii]
        im_path = os.path.join(im_root_dir, im_name)

        vis_im_name = 'im_{0}_hpe_vis_result.jpg'.format(ii)
        vis_im_path = os.path.join(vis_result_dir, vis_im_name)

        start_time = time.time()
        im = cv2.imread(im_path, 1)

        pose = single_image_testing_for_pose_on_lip_dataset(net, \
                                                            im, \
                                                            transform=transform, \
                                                            stride=stride, \
                                                            crop_size=crop_size, \
                                                            scale_multiplier=scale_multiplier, \
                                                            num_of_joints=num_of_joints, \
                                                            visualization=visualization, \
                                                            vis_im_path=vis_im_path)

        pose_list[ii, :, :] = pose[:, :]

        end_time = time.time()
        total_time += (end_time - start_time)

        print('Testing for pose on LIP dataset: [{0}/{1}], name: {2}, cur time: {3:.4f}, avg time: {4:.4f}'.format(ii + 1, num_of_im, im_name, (end_time - start_time), total_time / (ii + 1)))

    return pose_list 

def single_image_testing_for_pose_on_lip_dataset(net, \
                                                 im, \
                                                 transform=None, \
                                                 stride=4, \
                                                 crop_size=256, \
                                                 scale_multiplier=[1], \
                                                 num_of_joints=16, \
                                                 visualization=False, \
                                                 vis_im_path='exps/preds/vis_results/hpe_vis_result.jpg'):

    # Get the original image size
    im_height = im.shape[0]
    im_width = im.shape[1]
    long_edge = max(im_height, im_width)

    # Use the image center as the person center
    ori_center = np.array([[im_width / 2.0, im_height / 2.0]])
	
    # Resize the long edge of image to crop_size
    scale_provided = long_edge * 1.0 / crop_size 
    base_scale = 1.0 / scale_provided

    # Variables to store multi-scale test images and their crop parameters
    cropped_im_list = []
    cropped_param_list = [] 
    flipped_cropped_im_list = []

    for sm in scale_multiplier:
        # Resized image to base scales
        scale = base_scale * sm
        resized_im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_center = np.zeros([1, 2])
        scaled_center[0, 0] = int(ori_center[0, 0] * scale)
        scaled_center[0, 1] = int(ori_center[0, 1] * scale)

        # Get flipped images
        flipped_resized_im = cv2.flip(resized_im, 1)

        # Crop image for testing
        cropped_im, cropped_param = augmentation_cropped(resized_im, scaled_center, crop_x=crop_size, crop_y=crop_size, max_center_trans=0)
        cropped_im_list.append(cropped_im)
        cropped_param_list.append(cropped_param)

        # Get flipped cropped image for testing
        flipped_cropped_im = cv2.flip(cropped_im, 1)
        flipped_cropped_im_list.append(flipped_cropped_im)

    # Transform image
    input_im_list = []
    flipped_input_im_list = []
    if transform is not None:
        for cropped_im in cropped_im_list:
            input_im = transform(cropped_im)
            input_im_list.append(input_im)
        for flipped_cropped_im in flipped_cropped_im_list:
            flipped_input_im = transform(flipped_cropped_im)
            flipped_input_im_list.append(flipped_input_im)
    else:
        for cropped_im in cropped_im_list:
            input_im =cropped_im.copy()
            input_im_list.append(input_im)
        for flipped_cropped_im in flipped_cropped_im_list:
            flipped_input_im = flipped_cropped_im.copy()
            flipped_input_im_list.append(flipped_input_im)

    # Preparing input variable
    batch_input_im = input_im_list[0].view(-1, 3, crop_size, crop_size)    
    for smi in range(1, len(input_im_list)):
        batch_input_im = torch.cat((batch_input_im, input_im_list[smi].view(-1, 3, crop_size, crop_size)), 0)
    batch_input_im = batch_input_im.cuda(async=True)
    batch_input_var = torch.autograd.Variable(batch_input_im, volatile=True)

    # Preparing flipped input variable
    batch_flipped_input_im = flipped_input_im_list[0].view(-1, 3, crop_size, crop_size)
    for smi in range(1, len(flipped_input_im_list)):
        batch_flipped_input_im = torch.cat((batch_flipped_input_im, flipped_input_im_list[smi].view(-1, 3, crop_size, crop_size)), 0)
    batch_flipped_input_im = batch_flipped_input_im.cuda(async=True)
    batch_flipped_input_var = torch.autograd.Variable(batch_flipped_input_im, volatile=True)

    # Get predicted heatmaps and convert them to numpy array
    pose_output, parsing_output = net(batch_input_var)
    if isinstance(pose_output, list):
        pose_output = pose_output[-1]
    pose_output = pose_output.data
    pose_output = pose_output.cpu().numpy()

    # Get predicted flipped heatmaps and convert them to numpy array
    flipped_pose_output, flipped_parsing_output = net(batch_flipped_input_var)
    if isinstance(flipped_pose_output, list):
        flipped_pose_output = flipped_pose_output[-1]
    flipped_pose_output = flipped_pose_output.data
    flipped_pose_output = flipped_pose_output.cpu().numpy()
    
    # First fuse the original prediction with flipped prediction
    fused_pose_output = np.zeros((pose_output.shape[0], pose_output.shape[1] - 1, crop_size, crop_size))
    flipped_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15]
    for smi in range(0, len(scale_multiplier)):
        # Get single scale output
        single_scale_output = pose_output[smi, :, :, :].copy()
        single_scale_flipped_output = flipped_pose_output[smi, :, :, :].copy()

        # fuse each joint's heatmap
        for ji in range(0, num_of_joints):
            # Get the original heatmap
            heatmap = single_scale_output[ji, :, :].copy()
            heatmap = cv2.resize(heatmap, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)  
    
            # Get the flipped heatmap
            flipped_heatmap = single_scale_flipped_output[flipped_idx[ji], :, :].copy()
            flipped_heatmap = cv2.resize(flipped_heatmap, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            flipped_heatmap = cv2.flip(flipped_heatmap, 1)

            # Average the original heatmap with flipped heatmap
            heatmap += flipped_heatmap
            heatmap *= 0.5

            fused_pose_output[smi, ji, :, :] = heatmap

    # Second fuse multi-scale predictions
    ms_fused_pose_output = np.zeros((fused_pose_output.shape[1], crop_size, crop_size)) 
    for smi in range(0, len(scale_multiplier)):
        single_scale_output = fused_pose_output[smi, :, :, :]
        crop_param = cropped_param_list[smi]

        # Crop the heatmaps without padding
        cropped_single_scale_output = single_scale_output[:, crop_param[0, 3]:crop_param[0, 7], crop_param[0, 2]:crop_param[0, 6]]

        # Resize the cropped heatmaps to base scale
        scaled_single_scale_output = cropped_single_scale_output.transpose((1, 2, 0))
        scaled_single_scale_output = cv2.resize(scaled_single_scale_output, None, fx=1.0/scale_multiplier[smi], fy=1.0/scale_multiplier[smi], interpolation=cv2.INTER_LINEAR)        
        scaled_single_scale_output = scaled_single_scale_output.transpose((2, 0, 1))  
        
        # Cropping position
        ul_x = int((crop_size - scaled_single_scale_output.shape[2]) / 2.0)
        ul_y = int((crop_size - scaled_single_scale_output.shape[1]) / 2.0)
        br_x = ul_x + scaled_single_scale_output.shape[2]
        br_y = ul_y + scaled_single_scale_output.shape[1]

        # Paste to base-scale heatmaps
        ms_fused_pose_output[:, ul_y:br_y, ul_x:br_x] += scaled_single_scale_output

    # Normalize with number of scales
    ms_fused_pose_output = ms_fused_pose_output / len(scale_multiplier)
    
    pose = np.zeros((num_of_joints, 3))
    cropped_param = cropped_param_list[scale_multiplier.index(1)]
    for ji in range(0, num_of_joints):
        heatmap = ms_fused_pose_output[ji, :, :]
        heatmap = gaussian_filter(heatmap, sigma=3)

        pred_pos = np.unravel_index(heatmap.argmax(), np.shape(heatmap))
        pred_x = (pred_pos[1] - cropped_param[0, 2] + cropped_param[0, 0]) / base_scale
        pred_y = (pred_pos[0] - cropped_param[0, 3] + cropped_param[0, 1]) / base_scale

        pose[ji, 0] = pred_x
        pose[ji, 1] = pred_y
        pose[ji, 2] = heatmap[pred_pos[0], pred_pos[1]]   

    if visualization:
        vis_hpe_results(im, pose, save_im=True, save_path=vis_im_path)
    
    return pose 

## Save pose estimation results
def save_hpe_results_to_lip_format(im_name_list, pose_list, save_path='exps/preds/csv_results/pred_keypoints_lip.csv', eval_num=-1):
    
    if eval_num > 0:
        num_of_im = eval_num
    else:
        num_of_im = len(im_name_list)

    result_list = []
    idx_map_to_lip = [10, 9, 8, 11, 12, 13, 15, 14, 1, 0, 4, 3, 2, 5, 6, 7]
    for ii in range(0, num_of_im):
        single_result = []
        single_result.append(im_name_list[ii])
        for ji in range(0, len(idx_map_to_lip)):
            single_result.append(str(int(pose_list[ii, idx_map_to_lip[ji], 0])))
            single_result.append(str(int(pose_list[ii, idx_map_to_lip[ji], 1])))
        result_list.append(single_result)

    with open(save_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in result_list:
            writer.writerow(line)

def multi_image_testing_for_parsing_on_lip_dataset(net, \
                                                   im_root_dir, \
                                                   im_name_list, \
                                                   parsing_pred_dir='exps/preds/parsing_results', \
                                                   transform=None, \
                                                   crop_size=256, \
                                                   num_of_parts=20, \
                                                   visualization=False, \
                                                   vis_result_dir='exps/preds/vis_results', \
                                                   eval_num=-1):

    print('LIP Testing for parsing')

    if eval_num > 0:
        num_of_im = eval_num
    else:
        num_of_im = len(im_name_list)

    total_time = 0

    for ii in range(0, num_of_im):

        im_name = im_name_list[ii]
        im_path = os.path.join(im_root_dir, im_name)

        vis_im_name = 'im_{0}_parsing_vis_result.jpg'.format(ii)
        vis_im_path = os.path.join(vis_result_dir, vis_im_name)

        start_time = time.time()
        im = cv2.imread(im_path, 1)

        parsing = single_image_testing_for_parsing_on_lip_dataset(net, \
                                                                  im, \
                                                                  transform=transform, \
                                                                  crop_size=crop_size, \
                                                                  num_of_parts=num_of_parts, \
                                                                  visualization=visualization, \
                                                                  vis_im_path=vis_im_path)

        parsing_pred_path = os.path.join(parsing_pred_dir, im_name.split('.')[0] + '.png')
        cv2.imwrite(parsing_pred_path, parsing)

        end_time = time.time()
        total_time += (end_time - start_time)

        print('Testing for parsing on LIP dataset: [{0}/{1}], name: {2}, cur time: {3:.4f}, avg time: {4:.4f}'.format(ii + 1, num_of_im, im_name, (end_time - start_time), total_time / (ii + 1)))


def single_image_testing_for_parsing_on_lip_dataset(net, \
                                                    im, \
                                                    transform=None, 
                                                    crop_size=256, \
                                                    num_of_parts=20, \
                                                    visualization=False, \
                                                    vis_im_path='exps/preds/vis_results/parsing_vis_result.jpg'):
    
    # height, width and long edge of image
    im_height = im.shape[0]
    im_width = im.shape[1]
    long_edge = max(im_height, im_width)

    # Use the image center as the person center
    center = np.array([[im_width / 2.0, im_height / 2.0]])
    
    # Resize the long edge of image to crop_size
    scale_provided = long_edge * 1.0 / crop_size 
    scale = 1 / scale_provided
    resized_im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    center[0, 0] = int(center[0, 0] * scale)
    center[0, 1] = int(center[0, 1] * scale)

    # Crop image for testing
    cropped_im, cropped_param = augmentation_cropped(resized_im, center, crop_x=crop_size, crop_y=crop_size, max_center_trans=0)

    # Transform image
    if transform is not None:
        input_im = transform(cropped_im)
    else:
        input_im =cropped_im.copy()

    # Preparing input variable
    input_im = input_im.view(-1, 3, input_im.size(1), input_im.size(2))
    input_im = input_im.cuda(async=True)
    input_var = torch.autograd.Variable(input_im, volatile=True)

    # Get predicted heatmaps and convert them to numpy array
    pose_output, parsing_output = net(input_var)
    if isinstance(parsing_output, list):
        parsing_output = parsing_output[-1]
    parsing_output = parsing_output.data
    parsing_output = parsing_output.view(parsing_output.size(1), parsing_output.size(2), parsing_output.size(3))
    parsing_output = parsing_output.cpu().numpy()
    output_argmax = parsing_output.argmax(0)

    parsing = np.zeros((resized_im.shape[0], resized_im.shape[1]))
    parsing[cropped_param[0, 1]:cropped_param[0, 5], 
            cropped_param[0, 0]:cropped_param[0, 4]] = output_argmax[cropped_param[0, 3]:cropped_param[0, 7], 
                                                                     cropped_param[0, 2]:cropped_param[0, 6]]

    parsing = cv2.resize(parsing, dsize=(im_width, im_height), interpolation=cv2.INTER_NEAREST)

    if visualization:
        vis_parsing_results(im, parsing, stride=1, save_im=True, save_path=vis_im_path)

    return parsing 

if __name__ == '__main__':
    print('Testing MuLA models...')
