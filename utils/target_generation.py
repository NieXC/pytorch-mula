import os
import sys
import numpy as np
import random
import cv2

def gen_parsing_target(parsing_anno, scale_param=None, rotate_param=None, crop_param=None, flip_param=None,stride=8):
	
	parsing_target = parsing_anno.copy()
	
	if scale_param is not None:
		parsing_target = cv2.resize(parsing_target, None, fx=scale_param, fy=scale_param, interpolation=cv2.INTER_NEAREST)
	
	if rotate_param is not None:
		parsing_target = cv2.warpAffine(parsing_target, rotate_param[0], dsize=(int(rotate_param[1]), int(rotate_param[2])), 
		                               flags=cv2.INTER_NEAREST, 
		                               borderMode=cv2.BORDER_CONSTANT, 
		                               borderValue=(0, 0, 0))

	if crop_param is not None:
		temp_crop_parsing_target = np.zeros((crop_param[1], crop_param[2]))
		temp_crop_parsing_target[crop_param[0][0, 3]:crop_param[0][0, 7], crop_param[0][0, 2]:crop_param[0][0, 6]] = \
		                      parsing_target[crop_param[0][0, 1]:crop_param[0][0, 5], crop_param[0][0, 0]:crop_param[0][0, 4]]
		parsing_target = temp_crop_parsing_target.astype(np.uint8)

	if flip_param is not None:
		if flip_param:
			parsing_target = cv2.flip(parsing_target, 1)
			# Flip left and right parts
			# Right-arm: 15, Right-leg: 17, Right-shoe: 19
			# Left-arm: 14 , Left-leg: 16, Left-shoe: 18
			right_idx = [15, 17, 19]
			left_idx = [14, 16, 18]
			for i in range(0, 3):
				right_pos = np.where(parsing_target == right_idx[i])
				left_pos = np.where(parsing_target == left_idx[i])
				parsing_target[right_pos[0], right_pos[1]] = left_idx[i]
				parsing_target[left_pos[0], left_pos[1]] = right_idx[i]

	parsing_target = cv2.resize(parsing_target, None, fx=(1.0 / stride), fy=(1.0 / stride), interpolation=cv2.INTER_NEAREST)

	return parsing_target

def gen_pose_target(joints, visibility, stride=8, grid_x=46, grid_y=46, sigma=7):
    #print "Target generation -- Gaussian maps"

    joint_num = joints.shape[0]
    gaussian_maps = np.zeros((joint_num + 1, grid_y, grid_x))
    for ji in range(0, joint_num):
        if visibility[ji]:
            gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
            gaussian_maps[ji, :, :] = gaussian_map[:, :]

    # Get background heatmap
    max_heatmap = gaussian_maps.max(0)

    gaussian_maps[joint_num, :, :] = 1 - max_heatmap

    return gaussian_maps

def gen_single_gaussian_map(center, stride, grid_x, grid_y, sigma):
    #print "Target generation -- Single gaussian maps"

    gaussian_map = np.zeros((grid_y, grid_x))
    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    for g_y in range(start_y, end_y):
        for g_x in range(start_x, end_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            gaussian_map[g_y, g_x] += np.exp(-exponent)
            if gaussian_map[g_y, g_x] > 1:
                gaussian_map[g_y, g_x] = 1

    return gaussian_map
