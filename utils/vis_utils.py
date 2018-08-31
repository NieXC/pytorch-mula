import os
import sys
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
import math

joint_names = ["Head top", "Neck",
               "R shoulder", "R elbow", "R wrist",
               "L shoulder", "L elbow", "L wrist",
               "R hip", "R knee", "R ankle",
               "L hip", "L knee", "L ankle",
               "Thorax", "Pelvis",
               "Background"]

def vis_gaussian_maps(im, gaussian_maps, stride, save_im=False, save_path='exps/preds/vis_results/gaussian_map_on_im.jpg'):
    #print 'Visualize gaussian maps'

    gm_num = gaussian_maps.shape[0]
    plot_grid_size = np.ceil(np.sqrt(gm_num))
    for gmi in range(0, gm_num):
        gaussian_map = gaussian_maps[gmi, :, :].copy()
        if gaussian_map.max() > 0:
            gaussian_map -= gaussian_map.min()
            gaussian_map /= gaussian_map.max()
        resized_gaussian_map = gaussian_map * 255
        resized_gaussian_map = cv2.resize(resized_gaussian_map, None, fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
        resized_gaussian_map = resized_gaussian_map.astype(np.uint8)
        resized_gaussian_map = cv2.applyColorMap(resized_gaussian_map, cv2.COLORMAP_JET)
        vis_gaussian_map_im = cv2.addWeighted(resized_gaussian_map, 0.5, im.astype(np.uint8), 0.5, 0.0);

        plt.subplot(plot_grid_size, plot_grid_size, gmi + 1),plt.imshow(vis_gaussian_map_im[:, :, [2, 1, 0]]), plt.title(joint_names[gmi])
        plt.xticks([])
        plt.yticks([])
    if save_im:	
        plt.savefig(save_path)

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='exps/preds/vis_results/parsing_map_on_im.jpg'):

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0],  [255, 170, 0], 
				   [255, 0, 85], [255, 0, 170],
				   [0, 255, 0], [85, 255, 0], [170, 255, 0],
				   [0, 255, 85], [0, 255, 170],
				   [0, 0, 255], [85, 0, 255], [170, 0, 255],
				   [0, 85, 255], [0, 170, 255],
				   [255, 255, 0], [255, 255, 85], [255, 255, 170],
				   [255, 0, 255], [255, 85, 255], [255, 170, 255],
				   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
	
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

def vis_hpe_results(im, joints, save_im=False, save_path='exps/preds/vis_results/hpe_vis_result.jpg'):
	
    # Colors for all 16 joints
    joint_colors = [[255, 0, 0], [255, 85, 0],  [255, 170, 0], [255, 255, 0], 
		            [170, 255, 0], [85, 255, 0],  [0, 255, 0],   [0, 255, 85], 
		            [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
		            [0, 0, 255],  [85, 0, 255], [255, 0, 170], [255, 0, 85]]

    # Limb connections 
    limbs = [[1, 0],
			 [1, 2],   [2, 3],   [3, 4],
			 [1, 5],   [5, 6],   [6, 7],
			 [1, 14],  [14, 15],
			 [15, 8],  [8, 9],   [9, 10],
			 [15, 11], [11, 12], [12, 13]]

    # Number of joints and visualization image copy
    num_of_joints = joints.shape[0]
    vis_im = im.copy()

    # Draw limbs
    for li in range(0, len(limbs)):
        cv2.line(vis_im, (int(joints[limbs[li][0], 0]), int(joints[limbs[li][0], 1])), \
			             (int(joints[limbs[li][1], 0]), int(joints[limbs[li][1], 1])), \
			             (233, 161, 0), 3, cv2.LINE_AA)
    # Draw joints
    for ji in range(0, num_of_joints):
        cv2.circle(vis_im, (int(joints[ji, 0]), int(joints[ji, 1])), 3, joint_colors[ji], -1, cv2.LINE_AA)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def vis_parsing_results(im, parsing, stride, save_im=False, save_path='exps/preds/vis_results/parsing_vis_result.jpg'):

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0],  [255, 170, 0], 
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
