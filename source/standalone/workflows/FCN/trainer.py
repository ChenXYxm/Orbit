import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
class Trainer(object):
    def __init__(self, future_reward_discount):
        self.use_cuda = True
        self.model = reinforcement_net(self.use_cuda)
        
        self.future_reward_discount = future_reward_discount

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0
    def forward(self, img_observation, is_volatile=False, specific_rotation=-1):
        input_image = img_observation.astype(float)/255
        input_data = torch.from_numpy(input_image.astype(np.float32)).permute(0,3,1,2)
        output_prob, state_feat = self.model.forward(input_data, is_volatile, specific_rotation)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,:,:]
                stop_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,:,:]
            else:
                push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,:,:]), axis=0)
                stop_predictions = np.concatenate((stop_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,:,:]), axis=0)
        return push_predictions, stop_predictions, state_feat
    
    def get_label_value(self,reward, primitive_action, prev_push_predictions, prev_stop_predictions, next_image):
        current_reward = 0
        if primitive_action == 'push':
            current_reward = reward
        elif primitive_action == 'stop':
            current_reward = -1
        if current_reward<0:
            future_reward = 0
        else:
            next_push_predictions, next_stop_predictions, next_state_feat = self.forward(next_image, is_volatile=True)
            future_reward = max(np.max(next_push_predictions), 0)
        if primitive_action == 'push' and reward<0:
            expected_reward = self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
        else:
            expected_reward = current_reward + self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward
    # Compute labels and backpropagate
    def backprop(self, depth_heightmap, primitive_action, best_pix_ind, label_value):
        label = np.zeros((1,100,100))
        action_area = np.zeros((100,100))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((100,100))
        tmp_label[action_area > 0] = label_value
        label[0,:,:] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((100,100))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,:,:] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0
        if primitive_action == 'push':

            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, state_feat = self.forward(depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1,100,100), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1,100,100), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

        elif primitive_action == 'stop':

            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, state_feat = self.forward(depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1,100,100), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1,100,100), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

            push_predictions, grasp_predictions, state_feat = self.forward(depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1,100,100), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1,100,100), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value/2

        print('Training loss: %f' % (loss_value))
        self.optimizer.step()



