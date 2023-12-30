from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time

class reinforcement_net(nn.Module):

    def __init__(self, use_cuda=True): # , snapshot=None
        super().__init__()
        self.use_cuda = use_cuda
        self.num_rotations = 4
        self.pre_pushnet = nn.Sequential(OrderedDict([
            ('push-vgg0',nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)),
            ('push-vgg-relu0', nn.ReLU(inplace=True)),
            # ('push-vgg1',nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)),
            # ('push-vgg-relu1', nn.ReLU(inplace=True)),
            ('push-pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1, ceil_mode=False)),
            ('push-vgg2',nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)),
            ('push-vgg-relu2', nn.ReLU(inplace=True)),
            # ('push-vgg3',nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)),
            # ('push-vgg-relu3', nn.ReLU(inplace=True)),
            ('push-pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1, ceil_mode=False)),
            
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.pushnet = nn.Sequential(OrderedDict([
            
            ('push-norm0', nn.BatchNorm2d(128)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.pre_stopnet = nn.Sequential(OrderedDict([
            ('stop-vgg0',nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)),
            ('stop-vgg-relu0', nn.ReLU(inplace=True)),
            # ('stop-vgg1',nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)),
            # ('stop-vgg-relu1', nn.ReLU(inplace=True)),
            ('stop-pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1, ceil_mode=False)),
            ('stop-vgg2',nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)),
            ('stop-vgg-relu2', nn.ReLU(inplace=True)),
            # ('stop-vgg3',nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)),
            # ('stop-vgg-relu3', nn.ReLU(inplace=True)),
            ('stop-pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1, ceil_mode=False)),
            
        ]))
        self.stopnet = nn.Sequential(OrderedDict([
            
            ('stop-norm0', nn.BatchNorm2d(128)),
            ('stop-relu0', nn.ReLU(inplace=True)),
            ('stop-conv0', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('stop-norm1', nn.BatchNorm2d(64)),
            ('stop-relu1', nn.ReLU(inplace=True)),
            ('stop-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []
    def forward(self, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_depth_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    interm_push_depth_feat = self.pre_pushnet(rotate_depth)
                    interm_stop_depth_feat = self.pre_stopnet(rotate_depth)
                    
                    interm_feat.append([interm_push_depth_feat, interm_stop_depth_feat])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_depth_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_depth_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=4, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_depth_feat), flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=4, mode='bilinear').forward(F.grid_sample(self.stopnet(interm_stop_depth_feat), flow_grid_after, mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_depth_data.size())

            # Rotate images clockwise
            if self.use_cuda:
              rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
              rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_push_depth_feat = self.pre_pushnet(rotate_depth)
            interm_stop_depth_feat = self.pre_stopnet(rotate_depth)
            
            self.interm_feat.append([interm_push_depth_feat, interm_stop_depth_feat])
            

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_depth_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_depth_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=4, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_depth_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=4, mode='bilinear').forward(F.grid_sample(self.stopnet(interm_stop_depth_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat
'''
VGG16(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''