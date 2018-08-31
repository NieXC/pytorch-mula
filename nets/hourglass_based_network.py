import torch
import torch.nn as nn
import math
import time
import numpy as np

from nets.adaptive_conv import AdaptiveConv2d
from nets.network_init import GaussianInit, MSRAInit

# Pre-activation residual block
class ResidualBlock_PreAct(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ResidualBlock_PreAct, self).__init__()
		
        self.in_plane = in_plane
        self.out_plane = out_plane

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(in_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_plane, out_plane, 1, bias=False)	
        )
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_plane, int(out_plane / 2), 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), int(out_plane / 2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), out_plane, 1, bias=False)
    )

    def forward(self, x):

        out = self.res_block(x)

        residual = x
        if self.in_plane != self.out_plane:
	        residual = self.conv1x1(x)		
        out += residual

        return out

# Pose-activation residual block
class ResidualBlock_PostAct(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ResidualBlock_PostAct, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, 1, bias=False),
            nn.BatchNorm2d(out_plane)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(in_plane, int(out_plane / 2), 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), int(out_plane / 2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), out_plane, 1, bias=False),
            nn.BatchNorm2d(out_plane)
        )
        self.relu = nn.ReLU(inplace=True)

        self.in_plane = in_plane
        self.out_plane = out_plane

    def forward(self, x):	

        out = self.res_block(x)

        residual = x
        if self.in_plane != self.out_plane:
            residual = self.conv1x1(x)		

        out += residual
        out = self.relu(out)

        return out

# Hourglass block
class HourglassBlock(nn.Module):
    def __init__(self, num_of_feat=256, num_of_module=1):
        super(HourglassBlock, self).__init__()
		
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn = nn.BatchNorm2d(num_of_feat)
        self.relu = nn.ReLU(inplace=True)

        self.srb1 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb2 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb3 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb4 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb5 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb6 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb7 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb8 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb9 = self._make_seq_res_blocks(num_of_feat, num_of_module)

    # Construct sequence of residual blocks
    def _make_seq_res_blocks(self, num_of_feat, num_of_module):
        seq_res_blocks = []

        for i in range(num_of_module):
            seq_res_blocks.append(ResidualBlock_PreAct(num_of_feat, num_of_feat))

        return nn.Sequential(*seq_res_blocks)

    def forward(self, x):
        # Downsample process
        x1 = self.srb1(x)                        # 1/1
        x1_downsample = self.downsample(x1)      # 1/2
        x2 = self.srb2(x1_downsample)            # 1/2
        x2_downsample = self.downsample(x2)      # 1/4
        x3 = self.srb3(x2_downsample)            # 1/4
        x3_downsample = self.downsample(x3)      # 1/8
        x4 = self.srb4(x3_downsample)            # 1/8
        x4_downsample = self.downsample(x4)      # 1/16

        # Bottle neck
        bottle_neck = self.srb5(x4_downsample)   # 1/16

        # Upsample process
        x4_upsample = self.upsample(bottle_neck) # 1/8
        x4_sym = x4 + x4_upsample                # 1/8 
        x4_sym = self.srb6(x4_sym)               # 1/8
        x3_upsample = self.upsample(x4_sym)      # 1/4
        x3_sym = x3 + x3_upsample                # 1/4
        x3_sym = self.srb7(x3_sym)               # 1/4
        x2_upsample = self.upsample(x3_sym)      # 1/2
        x2_sym = x2 + x2_upsample                # 1/2
        x2_sym = self.srb8(x2_sym)               # 1/2
        x1_upsample = self.upsample(x2_sym)      # 1/1
        x1_sym = x1 + x1_upsample                # 1/1
        x1_sym = self.srb9(x1_sym)               # 1/1

        x1_sym = self.bn(x1_sym)
        x1_sym = self.relu(x1_sym)

        return x1_sym

# Hourglass network
class HourglassNetwork(nn.Module):
    def __init__(self, num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
        super(HourglassNetwork, self).__init__()
		
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False)
        )
		
        self.basic_block = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock_PostAct(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.num_of_stages = num_of_stages

        self.res_pre_act = ResidualBlock_PreAct(128, num_of_feat)

        self.bn = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(num_of_stages)]) 
        self.relu = nn.ReLU(inplace=True)

        self.hg_list = self._make_multi_stage_hg(num_of_feat, num_of_module, num_of_stages)
        self.conv1x1_1_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages)
        self.conv1x1_2_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages - 1)
        self.conv_pred_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_class, num_of_stages)
        self.conv_remap_list = self._make_multi_stage_conv1x1(num_of_class, num_of_feat, num_of_stages - 1)

    def _make_multi_stage_hg(self, num_of_feat, num_of_module, num_of_stages):
		
        hg_list = nn.ModuleList([HourglassBlock(num_of_feat, num_of_module) for i in range(num_of_stages)])
		
        return hg_list
    
    def _make_multi_stage_conv1x1(self, in_num_of_feat, out_num_of_feat, num_of_stages):

        conv1x1_list = nn.ModuleList([nn.Conv2d(in_num_of_feat, out_num_of_feat, 1, 1) for i in range(num_of_stages)])

        return conv1x1_list

    def forward(self, x):

        pred_list = []

        pool = self.basic_block(x)
        residual = pool
        out = self.res_block(pool)
        out += residual
        feat = self.res_pre_act(out)

        feat_to_next_stage = feat
        for i in range(self.num_of_stages):
			
            hg_out = self.hg_list[i](feat_to_next_stage)
			
            feat_1 = self.conv1x1_1_list[i](hg_out)
            feat_1 = self.bn[i](feat_1)
            feat_1 = self.relu(feat_1)
			
            pred = self.conv_pred_list[i](feat_1)
            pred_list.append(pred)
			
            if i < self.num_of_stages - 1:
                feat_2 = self.conv1x1_2_list[i](feat_1)
                feat_remap = self.conv_remap_list[i](pred)
                feat_to_next_stage = feat_to_next_stage + feat_2 + feat_remap

        return pred_list

# MuLA with Hourglass network as backbone 
class mula_hg_based_network(nn.Module):
    def __init__(self, stage_num_of_encoder=5, module_num_of_pose_encoder=1, num_of_joint=16, module_num_of_parsing_encoder=1, num_of_part=20, num_of_feat=256):
        super(mula_hg_based_network, self).__init__()

        # Pose network 
        self.pose_encoder = HourglassNetwork(num_of_feat=num_of_feat, \
                                             num_of_class=num_of_joint + 1, \
                                             num_of_module=module_num_of_pose_encoder, \
                                             num_of_stages=stage_num_of_encoder)
        self.pose_classifier = self.pose_encoder.conv_pred_list
		
        # Parsing network
        self.parsing_encoder = HourglassNetwork(num_of_feat=num_of_feat, \
                                                num_of_class=num_of_part, \
                                                num_of_module=module_num_of_parsing_encoder, \
                                                num_of_stages=stage_num_of_encoder)
        self.parsing_classifier = self.parsing_encoder.conv_pred_list

        # Parameter adapter
        self.pose_param_adapter_list = self._make_param_adapter_list(num_of_feat, num_of_feat, stage_num_of_encoder)
        self.parsing_param_adapter_list = self._make_param_adapter_list(num_of_feat, num_of_feat, stage_num_of_encoder)

        # Parameter factorization 
        self.pose_conv1x1_U_list = self._make_conv1x1_list(num_of_feat, num_of_feat, stage_num_of_encoder)
        self.pose_conv1x1_V_list = self._make_conv1x1_list(num_of_feat, num_of_feat, stage_num_of_encoder)
        self.parsing_conv1x1_U_list = self._make_conv1x1_list(num_of_feat, num_of_feat, stage_num_of_encoder)
        self.parsing_conv1x1_V_list = self._make_conv1x1_list(num_of_feat, num_of_feat, stage_num_of_encoder)

        # Common components 
        self.pose_bn_list = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(stage_num_of_encoder)])
        self.parsing_bn_list = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(stage_num_of_encoder)])
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.log_softmax = nn.LogSoftmax()

        self.stage_num_of_encoder = stage_num_of_encoder

    def _make_conv1x1_list(self, num_of_feat_in, num_of_feat_out, num_of_conv1x1):
        
        conv1x1_list = nn.ModuleList([])

        for i in range(num_of_conv1x1):
            conv1x1 = nn.Conv2d(num_of_feat_in, num_of_feat_out, 1, 1)
            conv1x1_list.append(conv1x1)

        return conv1x1_list

    def _make_param_adapter_list(self, num_of_feat_in, num_of_feat_out, num_of_param_adapter):

        param_adapter_list = nn.ModuleList([])

        for i in range(num_of_param_adapter):
            param_adapter = nn.Sequential(
                            nn.Conv2d(num_of_feat_in, num_of_feat_out, 3, 2),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(num_of_feat_out, num_of_feat_out, 3, padding=1),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(num_of_feat_out, num_of_feat_out, 3, padding=1))
            param_adapter_list.append(param_adapter) 

        return param_adapter_list

    def forward(self, x):

        # Estimation results  
        pose_pred_list = []
        parsing_pred_list = []

        # For pose network
        pose_x = self.pose_encoder.basic_block(x)
        pose_x_res = pose_x
        pose_out = self.pose_encoder.res_block(pose_x)
        pose_out += pose_x_res
        pose_feat = self.pose_encoder.res_pre_act(pose_out)        
        pose_feat_to_next_stage = pose_feat

        # For parsing network  
        parsing_x = self.parsing_encoder.basic_block(x)
        parsing_x_res = parsing_x
        parsing_out = self.parsing_encoder.res_block(parsing_x)
        parsing_out += parsing_x_res
        parsing_feat = self.parsing_encoder.res_pre_act(parsing_out)
        parsing_feat_to_next_stage = parsing_feat 

        # Dynamic convolution kernels from parameter adapter 
        
        for i in range(self.stage_num_of_encoder):
            # Extract pose feature 
            pose_feat = self.pose_encoder.hg_list[i](pose_feat_to_next_stage)
            pose_feat = self.pose_encoder.conv1x1_1_list[i](pose_feat)
            pose_feat = self.pose_encoder.bn[i](pose_feat)
            pose_feat = self.relu(pose_feat)

            # Extract parsing feature 
            parsing_feat = self.parsing_encoder.hg_list[i](parsing_feat_to_next_stage)
            parsing_feat = self.parsing_encoder.conv1x1_1_list[i](parsing_feat)
            parsing_feat = self.parsing_encoder.bn[i](parsing_feat)
            parsing_feat = self.relu(parsing_feat)
		            
            # Mutual Adaptation 
            pose_theta_prime = self.pose_param_adapter_list[i](parsing_feat)
            pose_feat_res = self.pose_conv1x1_U_list[i](pose_feat)
            pose_adaptive_conv = AdaptiveConv2d(pose_feat_res.size(0) * pose_feat_res.size(1),
                                                pose_feat_res.size(0) * pose_feat_res.size(1),
                                                7, padding=3,
                                                groups=pose_feat_res.size(0) * pose_feat_res.size(1),
                                                bias=False)
            pose_feat_res = pose_adaptive_conv(pose_feat_res, pose_theta_prime)
            pose_feat_res = self.pose_conv1x1_V_list[i](pose_feat_res)
            pose_feat_res = self.pose_bn_list[i](pose_feat_res)
            pose_feat_res = self.relu(pose_feat_res)
            pose_feat_refined = pose_feat +  pose_feat_res

            parsing_phi_prime = self.parsing_param_adapter_list[i](pose_feat)
            parsing_feat_res = self.parsing_conv1x1_U_list[i](parsing_feat)
            parsing_adaptive_conv = AdaptiveConv2d(parsing_feat_res.size(0) * parsing_feat_res.size(1),
                                                   parsing_feat_res.size(0) * parsing_feat_res.size(1),
                                                   7, padding=3,
                                                   groups=parsing_feat_res.size(0) * parsing_feat_res.size(1),
                                                   bias=False)
            parsing_feat_res = parsing_adaptive_conv(parsing_feat_res, parsing_phi_prime)
            parsing_feat_res = self.parsing_conv1x1_V_list[i](parsing_feat_res)
            parsing_feat_res = self.parsing_bn_list[i](parsing_feat_res)
            parsing_feat_res = self.relu(parsing_feat_res)
            parsing_feat_refined = parsing_feat +  parsing_feat_res
    
            # Prediction
            pose_pred = self.pose_classifier[i](pose_feat_refined)
            pose_pred_list.append(pose_pred)
            parsing_pred = self.parsing_classifier[i](parsing_feat_refined)
            parsing_pred_list.append(self.log_softmax(self.upsample(parsing_pred)))
			
            if i < self.stage_num_of_encoder - 1:
                pose_feat_2 = self.pose_encoder.conv1x1_2_list[i](pose_feat)
                pose_feat_remap = self.pose_encoder.conv_remap_list[i](pose_pred)
                pose_feat_to_next_stage = pose_feat_to_next_stage + pose_feat_2 + pose_feat_remap

                parsing_feat_2 = self.parsing_encoder.conv1x1_2_list[i](parsing_feat)
                parsing_feat_remap = self.parsing_encoder.conv_remap_list[i](parsing_pred)
                parsing_feat_to_next_stage = parsing_feat_to_next_stage + parsing_feat_2 + parsing_feat_remap

        return pose_pred_list, parsing_pred_list

def HG_with_MSRAInit(num_of_feat=256, num_of_class=16, num_of_module=1, num_of_stages=8):
    model = MSRAInit(HourglassNetwork(num_of_feat=num_of_feat, 
									  num_of_class=num_of_class, 
									  num_of_module=num_of_module, 
									  num_of_stages=num_of_stages))
    return model

def HG_with_GaussianInit(num_of_feat=256, num_of_class=16, num_of_module=1, num_of_stages=8):
    model = GaussianInit(HourglassNetwork(num_of_feat=num_of_feat, 
										  num_of_class=num_of_class, 
										  num_of_module=num_of_module, 
										  num_of_stages=num_of_stages))
    return model

def MuLA_HG_MSRAInit(stage_num_of_encoder=5, module_num_of_pose_encoder=1, num_of_joint=16, module_num_of_parsing_encoder=1, num_of_part=20, num_of_feat=256):
    model = MSRAInit(mula_hg_based_network(stage_num_of_encoder=stage_num_of_encoder,
                                           module_num_of_pose_encoder=module_num_of_pose_encoder,
                                           num_of_joint=num_of_joint,
                                           module_num_of_parsing_encoder=module_num_of_parsing_encoder,
                                           num_of_part=num_of_part,
                                           num_of_feat=num_of_feat))
    return model

def MuLA_HG_GaussianInit(stage_num_of_encoder=5, module_num_of_pose_encoder=1, num_of_joint=16, module_num_of_parsing_encoder=1, num_of_part=20, num_of_feat=256):
    model = GaussianInit(mula_hg_based_network(stage_num_of_encoder=stage_num_of_encoder,
                                               module_num_of_pose_encoder=module_num_of_pose_encoder,
                                               num_of_joint=num_of_joint,
                                               module_num_of_parsing_encoder=module_num_of_parsing_encoder,
                                               num_of_part=num_of_part,
                                               num_of_feat=num_of_feat))
    return model

if __name__ == '__main__':
    print("MuLA-Hourglass based network")
