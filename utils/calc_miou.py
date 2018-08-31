import os
import numpy as np
from PIL import Image
import json

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(im_name_list, gt_root, pred_root, eval_num):
	num_of_class = 20

	hist = np.zeros((num_of_class, num_of_class))

	for ii in range(0, eval_num):
		
		file_name = im_name_list[ii].split('.')[0] + '.png'
		pred_path = os.path.join(pred_root, file_name)
		gt_path = os.path.join(gt_root, file_name)

		pred = Image.open(pred_path)
		pred_array = np.array(pred, dtype=np.int32)
		gt = Image.open(gt_path)
		gt_array = np.array(gt, dtype=np.int32)

		pred_shape = pred_array.shape
		gt_shape = gt_array.shape
		if not pred_shape == gt_shape:
			pred = pred.resize((gt_shape[1], gt_shape[0]), Image.ANTIALIAS)
			pred_array = np.array(pred, dtype=np.int32)

		hist += fast_hist(gt_array, pred_array, num_of_class)

	return hist

def calc_miou_lip_dataset(im_name_list, gt_root, pred_root, eval_num=-1):

    if eval_num <= 0:
        eval_num = len(im_name_list) 

    hist = compute_hist(im_name_list, gt_root, pred_root, eval_num)
    class_name = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
	              'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
	              'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
	              'rightShoe']

    # Num of correct pixels
    num_cor_pix = np.diag(hist)

    # Num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)

    # Evaluation 1: overall accuracy
    pixel_acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'pixel accuracy', pixel_acc)
    print('-' * 50)

    # Evaluation 2: mean accuracy & per-class accuracy 
    print('Accuracy for each class (pixel accuracy):')
    per_class_acc = num_cor_pix / num_gt_pix
    mean_acc = np.nanmean(per_class_acc)
    for i in range(20):
        print('%-15s: %f' % (class_name[i], per_class_acc[i])) 
    print('>>>', 'mean accuracy', mean_acc)
    print('-' * 50)

    # Evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    per_class_iou = num_cor_pix / union
    mean_iou = np.nanmean(per_class_iou)
    for i in range(20):
        print('%-15s: %f' % (class_name[i], per_class_iou[i]))
    print('>>>', 'mean IoU', mean_iou)
    print('-' * 50)

    # Evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    freq_w_iou = (freq[freq > 0] * per_class_iou[freq > 0]).sum()
    print('>>>', 'fwavacc', freq_w_iou)
    print('=' * 50)

    eval_result = {}
    eval_result['pixel_acc'] = pixel_acc
    eval_result['per_class_acc'] = per_class_acc
    eval_result['mean_acc'] = mean_acc
    eval_result['per_class_iou'] = per_class_iou
    eval_result['mean_iou'] = mean_iou
    eval_result['freq_w_iou'] = freq_w_iou

    return eval_result

if __name__ == '__main__':
    print('Calculate mIOU')
