import torch
import torch.nn as nn
import math
import time

from nets.adaptive_conv import AdaptiveConv2d
from nets.network_init import GaussianInit, MSRAInit

class VGGNetwork(nn.Module):
    def __init__(self, cfg='VGG16', batch_norm=False):
        super(VGGNetwork, self).__init__()

        self.layers = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 512],                                         
                       'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],                                
                       'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],                 
                       'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512]}

        self.model = self._make_layers(self.layers[cfg], batch_norm=batch_norm)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.model(x)
        return feat

class mula_vgg_based_network(nn.Module):
    def __init__(self, pose_encoder_cfg='VGG16', num_of_joint=16, parsing_encoder_cfg='VGG16', num_of_part=20, batch_norm=False, num_of_feat=512):
        super(mula_vgg_based_network, self).__init__()

        # Pose network
        self.pose_encoder = VGGNetwork(cfg=pose_encoder_cfg, batch_norm=batch_norm)
        self.pose_classifier = nn.Conv2d(num_of_feat, num_of_joint + 1, 1, 1) 

        # Parsing network
        self.parsing_encoder = VGGNetwork(cfg=parsing_encoder_cfg, batch_norm=batch_norm)
        self.parsing_classifier = nn.Conv2d(num_of_feat, num_of_part, 1, 1)

        # Parameter adapter
        self.pose_param_adapter = nn.Sequential(
                                  nn.Conv2d(num_of_feat, num_of_feat, 3, 2),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(num_of_feat, num_of_feat, 3, padding=1),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(num_of_feat, num_of_feat, 3, padding=1))

        self.parsing_param_adapter = nn.Sequential(
                                     nn.Conv2d(num_of_feat, num_of_feat, 3, 2),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(num_of_feat, num_of_feat, 3, padding=1),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(num_of_feat, num_of_feat, 3, padding=1))

        # Parameter factorization
        self.pose_conv1x1_U = nn.Conv2d(num_of_feat, num_of_feat, 1, 1)
        self.pose_conv1x1_V = nn.Conv2d(num_of_feat, num_of_feat, 1, 1)
        self.parsing_conv1x1_U = nn.Conv2d(num_of_feat, num_of_feat, 1, 1)
        self.parsing_conv1x1_V = nn.Conv2d(num_of_feat, num_of_feat, 1, 1)

        # Common components
        self.batch_norm = batch_norm
        self.pose_bn = nn.BatchNorm2d(num_of_feat)
        self.parsing_bn = nn.BatchNorm2d(num_of_feat)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        
        # For pose network
        pose_feat = self.pose_encoder(x)

        # For parsing network
        parsing_feat = self.parsing_encoder(x)

        # Mutual Adaptation 
        pose_theta_prime = self.pose_param_adapter(parsing_feat)
        pose_feat_res = self.pose_conv1x1_U(pose_feat)
        pose_adaptive_conv = AdaptiveConv2d(pose_feat_res.size(0) * pose_feat_res.size(1), 
                                            pose_feat_res.size(0) * pose_feat_res.size(1), 
                                            5, padding=1, 
                                            groups=pose_feat_res.size(0) * pose_feat_res.size(1), 
                                            bias=False)
        pose_feat_res = pose_adaptive_conv(pose_feat_res, pose_theta_prime)
        pose_feat_res = self.pose_conv1x1_V(pose_feat_res)
        if self.batch_norm:
            pose_feat_res = self.pose_bn(pose_feat_res)
        pose_feat_res = self.relu(pose_feat_res)
        pose_feat_refined = pose_feat + pose_feat_res
        
        parsing_phi_prime = self.parsing_param_adapter(pose_feat)
        parsing_feat_res = self.parsing_conv1x1_U(parsing_feat)
        parsing_adaptive_conv = AdaptiveConv2d(parsing_feat_res.size(0) * parsing_feat_res.size(1), 
                                               parsing_feat_res.size(0) * parsing_feat_res.size(1), 
                                               5, padding=1, 
                                               groups=parsing_feat_res.size(0) * parsing_feat_res.size(1), 
                                               bias=False)
        parsing_feat_res = parsing_adaptive_conv(parsing_feat_res, parsing_phi_prime)
        parsing_feat_res = self.parsing_conv1x1_V(parsing_feat_res)
        if self.batch_norm:
            parsing_feat_res = self.parsing_bn(parsing_feat_res)
        parsing_feat_res = self.relu(parsing_feat_res)
        parsing_feat_refined = parsing_feat + parsing_feat_res

        pose_pred = self.pose_classifier(pose_feat_refined)
        parsing_pred = self.log_softmax(self.upsample(self.parsing_classifier(parsing_feat_refined))) 

        return pose_pred, parsing_pred

def VGG_with_MSRAInit(cfg='VGG16', batch_norm=False):
    model = MSRAInit(VGGNetwork(cfg=cfg, batch_norm=batch_norm))
    return model

def VGG_with_GaussianInit(cfg='VGG16', batch_norm=False):
    model = GaussianInit(VGGNetwork(cfg=cfg, batch_norm=batch_norm))
    return model

def MuLA_VGG_MSRAInit(pose_encoder_cfg='VGG16', num_of_joint=16, parsing_encoder_cfg='VGG16', num_of_part=20, batch_norm=False, num_of_feat=512):
    model = MSRAInit(mula_vgg_based_network(pose_encoder_cfg=pose_encoder_cfg, 
                                            num_of_joint=num_of_joint,
                                            parsing_encoder_cfg=parsing_encoder_cfg,
                                            num_of_part=num_of_part,
                                            batch_norm=batch_norm,
                                            num_of_feat=num_of_feat))
    return model

def MuLA_VGG_GaussianInit(pose_encoder_cfg='VGG16', num_of_joint=16, parsing_encoder_cfg='VGG16', num_of_part=20, batch_norm=False, num_of_feat=512):
    model = GaussianInit(mula_vgg_based_network(pose_encoder_cfg=pose_encoder_cfg, 
                                                num_of_joint=num_of_joint,
                                                parsing_encoder_cfg=parsing_encoder_cfg,
                                                num_of_part=num_of_part,
                                                batch_norm=batch_norm,
                                                num_of_feat=num_of_feat))
    return model

if __name__ == '__main__':
	print("MuLA-VGG based network")
