import numpy as np
import csv
import scipy.io as sio
import math

def read_data(path, additional_dim):
    labels = []
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            label = row[1:]
            for l in range(len(label)):
                if label[l] == 'nan':
                    label[l] = '-1'
                label[l] = float(label[l])
            labels.append(label)

    data = np.array(labels)
    dim =2
    if additional_dim:
        dim = 3
    data = np.reshape(data,[data.shape[0], int(data.shape[1] / dim), dim])

    vis_label = np.zeros((data.shape[0], data.shape[1]))

    if additional_dim:
        vis_label[:, :] = data[:, :, 2]
        data = data[:, :, 0:2]
    else:
        vis_label = vis_label + 1
        data[data<0] = 1

    return data, vis_label

def get_head_size(gt):
	head_size = np.linalg.norm(gt[:,9,:] - gt[:,8,:], axis=1)
	for n in range(gt.shape[0]):
		if gt[n,8,0] < 0 or gt[n,9,0] < 0:  
			head_size[n] = 0

	return head_size

def get_norm_dist(pred, gt, ref_dist):
	N = pred.shape[0]
	P = pred.shape[1]
	dist = np.zeros([N, P])
	for n in range(N):
		cur_ref_dist = ref_dist[n]
		if cur_ref_dist == 0:
			dist[n, :] = -1   
		else:
			dist[n, :] = np.linalg.norm(gt[n, :, :] - pred[n, :, :], axis=1) / cur_ref_dist
			for p in range(P):
				if gt[n, p, 0] < 0 or gt[n, p, 1] < 0:
					dist[n, p] = -1
	return dist

def compute_pck(dist, pck_th_range):
	P = dist.shape[1]
	pck = np.zeros([len(pck_th_range), P + 2])

    # For individual joint
	for p in range(P):
		for thi in range(len(pck_th_range)):
			th = pck_th_range[thi]
			joint_dist = dist[:, p]
			pck[thi, p] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

	# For uppper body
	for thi in range(len(pck_th_range)):
		th = pck_th_range[thi]
		joint_dist = dist[:, 8:16]
		pck[thi, P] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

	# For all joints
	for thi in range(len(pck_th_range)):
		th = pck_th_range[thi]
		joints_index = list(range(0,6)) + list(range(8,16))
		joint_dist = dist[:, joints_index]
		pck[thi, P + 1] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

	return pck

def pck_table_output_lip_dataset(pck, method_name):
    str_template = '{0:10} & {1:6} & {2:6} & {3:6} & {4:6} & {5:6} & {6:6} & {7:6} & {8:6} & {9:6}'
    head_str = str_template.format('PCKh@0.5', 'Head', 'Sho.', 'Elb.', 'Wri.', 'Hip', 'Knee', 'Ank.', 'U.Body', 'Avg.')
    num_str = str_template.format(method_name, '%1.1f'%((pck[8]  + pck[9])  / 2.0),
                                               '%1.1f'%((pck[12] + pck[13]) / 2.0),
                                               '%1.1f'%((pck[11] + pck[14]) / 2.0),
                                               '%1.1f'%((pck[10] + pck[15]) / 2.0),
                                               '%1.1f'%((pck[2]  + pck[3])  / 2.0),
                                               '%1.1f'%((pck[1]  + pck[4])  / 2.0),
                                               '%1.1f'%((pck[0]  + pck[5])  / 2.0),
                                               '%1.1f'%(pck[-2]),
                                               '%1.1f'%(pck[-1]))
    print(head_str)
    print(num_str)

def calc_pck_lip_dataset(gt_path, pred_path, method_name='Ours', eval_num=5000):

    # Read prediction results
	pred, pred_vis_label = read_data(pred_path, False)
	pred = pred[0:eval_num, :, :]

    # Read groundtruth
	gt, gt_vis_label = read_data(gt_path, True)
	gt = gt[0:eval_num, :, :]

    # Make the pred and gt be the same shape
	assert gt.shape[0] == pred.shape[0], 'sample not matched'
	assert gt.shape[1] == pred.shape[1], 'joints not matched'
	assert gt.shape[2] == pred.shape[2], 'dim not matched'

    # PCK threshold range. 
	pck_th_range = np.array([0.50])

    # Get the reference distance for normalization
	ref_dist = get_head_size(gt)

    # Get the normalized distance between prediction and groundtruth
	dist = get_norm_dist(pred, gt, ref_dist)

	pck = compute_pck(dist, pck_th_range)
	pck_table_output_lip_dataset(pck[-1], method_name)

	return pck

if __name__ == "__main__":
    print('Calculate PCKh@0.5')	
