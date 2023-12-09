import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
from fcn2 import FCNet
push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True).cuda()
input_depth_data = torch.zeros((1,3,100,100)).cuda()
input_depth_data[0,:,20:40,20:40] = 1
print(np.argmax(input_depth_data.cpu().numpy()))
print(np.unravel_index(np.argmax(input_depth_data.cpu().numpy()), input_depth_data.cpu().numpy().shape))
interm_push_color_feat = push_color_trunk.features(input_depth_data)
model = FCNet(3,3).cuda()
interm_push_color_feat2,_ = model.forward(input_depth_data)
print(interm_push_color_feat2.size())
plt.imshow(interm_push_color_feat2[0,1].detach().cpu().numpy())
plt.show()
print(interm_push_color_feat.size())
plt.imshow(interm_push_color_feat[0,1].detach().cpu().numpy())
plt.show()
rotate_theta = np.radians(1*(360/16))
affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
affine_mat_before.shape = (2,3,1)
affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), (1,3,40,40))
rotate_color = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
rotate_color_post = torch.nn.Upsample(scale_factor=20, mode='bilinear').forward(interm_push_color_feat)
print(flow_grid_before.size())
print(rotate_color.size())
print(rotate_color_post.size())
plt.imshow(input_depth_data[0,1].cpu().numpy())
plt.show()
plt.imshow(rotate_color[0,1].cpu().numpy())
plt.show()
plt.imshow(rotate_color_post[0,1].detach().cpu().numpy())
plt.show()