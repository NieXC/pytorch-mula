import matplotlib
matplotlib.use('Agg')

import torch.utils.data as data
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import cv2
import numpy as np
import os
import os.path
import json
import random
import time

import utils.data_augmentation as data_aug
import utils.joint_transformation as joint_trans
import utils.target_generation as target_gen
import utils.vis_utils as vis_utils

# Use PIL to load image
def pil_loader(path):
    return Image.open(path).convert('RGB')

# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)

# LIP dataset Pose and Parsing
class LIPDataset(data.Dataset):
    def __init__(self, im_root, pose_anno_file, parsing_anno_root, transform=None, 
                                                                   target_transform=None, 
                                                                   loader=opencv_loader, 
                                                                   pose_net_stride=4, 
                                                                   sigma=7,
                                                                   parsing_net_stride=1,
                                                                   crop_size=256,
                                                                   target_dist=1.171, scale_min=0.7, scale_max=1.3,
                                                                   max_rotate_degree=40,
                                                                   max_center_trans=40,
                                                                   flip_prob=0.5,
                                                                   is_visualization=False):

        # Load train json file
        print('Loading training json file: {0}...'.format(pose_anno_file))
        train_list = []
        with open(pose_anno_file) as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            train_list = train_list + data_this
        print('Finished loading training json file')

        # Hyper-parameters
        self.im_root = im_root
        self.parsing_anno_root = parsing_anno_root
        self.pose_anno_list = train_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.pose_net_stride = pose_net_stride
        self.sigma = sigma
        self.parsing_net_stride = parsing_net_stride
        self.crop_size = crop_size
        self.target_dist = target_dist
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_rotate_degree = max_rotate_degree
        self.max_center_trans = max_center_trans
        self.flip_prob = flip_prob
        self.is_visualization = is_visualization

        # Number of train samples
        self.N_train = len(self.pose_anno_list)
    
    def __getitem__(self, index):
        # Select a training sample
        train_item = self.pose_anno_list[index]

        # Load training image
        im_name = train_item['im_name']
        im = self.loader(os.path.join(self.im_root, im_name))

        # Get parsing annotation 
        name_prefix = im_name.split('.')[0]
        parsing_anno_name = name_prefix + '.png'
        parsing_anno_path = os.path.join(self.parsing_anno_root, parsing_anno_name)
        parsing_anno = cv2.imread(parsing_anno_path, 0)

        # Get pose annotation 
        joints_all_info = np.array(train_item['joints'])
        joints_loc = np.zeros((joints_all_info.shape[0], 2))
        joints_loc[:, :] = joints_all_info[:, 0:2]

        # Reorder joints from MPI to ours
        joints_loc = joint_trans.transform_mpi_to_ours(joints_loc)

        # Get visibility of joints (The visibility information provided by the annotation is not accurate)
        coord_sum = np.sum(joints_loc, axis=1)
        visibility = coord_sum != 0

        # Get person center and scale
        person_center = np.array([train_item['objpos']])
        scale_provided = train_item['scale_provided']

        # Random scaling
        scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist, scale_min=self.scale_min, scale_max=self.scale_max)
        scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)

        # Random rotating
        rotated_im, rotate_param = data_aug.augmentation_rotate(scaled_im, max_rotate_degree=self.max_rotate_degree)
        rotated_joints, rotated_center = joint_trans.rotate_coords(scaled_joints, scaled_center, rotate_param)

        # Random cropping
        cropped_im, crop_param = data_aug.augmentation_cropped(rotated_im, rotated_center, crop_x=self.crop_size, crop_y=self.crop_size, max_center_trans=self.max_center_trans)
        cropped_joints, cropped_center = joint_trans.crop_coords(rotated_joints, rotated_center, crop_param)
        
        # Random flipping
        flipped_im, flip_param = data_aug.augmentation_flip(cropped_im, flip_prob=self.flip_prob)
        flipped_joints, flipped_center = joint_trans.flip_coords(cropped_joints, cropped_center, flip_param, flipped_im.shape[1])

        # If flip, then swap the visibility of left and right joints
        if flip_param:
            right_idx = [2, 3, 4, 8, 9, 10]
            left_idx = [5, 6, 7, 11, 12, 13]
            for i in range(0, 6):
                temp_visibility = visibility[right_idx[i]]
                visibility[right_idx[i]] = visibility[left_idx[i]]
                visibility[left_idx[i]] = temp_visibility

        # Generate pose target maps
        grid_x = flipped_im.shape[1] / self.pose_net_stride
        grid_y = flipped_im.shape[0] / self.pose_net_stride
        pose_target = target_gen.gen_pose_target(flipped_joints, visibility, self.pose_net_stride, grid_x, grid_y, self.sigma)

        # Generate parsing target maps 
        parsing_target = target_gen.gen_parsing_target(parsing_anno, 
                                                       scale_param=scale_param, 
                                                       rotate_param=[rotate_param, rotated_im.shape[1], rotated_im.shape[0]],
                                                       crop_param=[crop_param, cropped_im.shape[1], cropped_im.shape[0]],
                                                       flip_param=flip_param,
                                                       stride=self.parsing_net_stride)

        # Transform
        if self.transform is not None:
            aug_im = self.transform(flipped_im)
        else:
            aug_im = flipped_im
        
        # Visualize target maps
        if self.is_visualization:
            print('Visualize pose targets')
            vis_utils.vis_gaussian_maps(flipped_im, pose_target, self.pose_net_stride, save_im=True)
            vis_utils.vis_parsing_maps(flipped_im, parsing_target, self.parsing_net_stride, save_im=True)

        return aug_im, pose_target, parsing_target
    
    def __len__(self):
        return self.N_train

if __name__ == '__main__':
    print('Data loader for Human Pose Estimation with Parsing Induced Learner on LIP dataset')

    

