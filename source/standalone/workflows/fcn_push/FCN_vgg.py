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
                # Initialize network trunks with DenseNet pre-trained on ImageNet
        # self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        
        self.num_rotations = 16
        