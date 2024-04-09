import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Modules import ReplayBuffer, Transition, simple_Transition,MULTIDISCRETE_RESNET_Rotate
import numpy as np
import pickle
import random
import copy
import math
from collections import deque, defaultdict
import time
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def rotate_45(matrix,degree=45):
    # Get dimensions of the matrix
    rows, cols = matrix.shape

    # Create a rotated matrix with zeros
    rotated_matrix = np.zeros((rows, cols), dtype=matrix.dtype)

    # Rotate by -45 degrees (to simulate a 45-degree clockwise rotation)
    rotated = rotate(matrix, -degree, reshape=False, order=1, mode='constant', cval=0, prefilter=False)

    # Extract the center part of the rotated matrix
    center = (rotated.shape[0] - rows) // 2
    rotated_matrix = rotated[center:center + rows, center:center + cols]
    rotated_matrix[np.where(rotated_matrix<0.5)] = 0
    rotated_matrix[np.where(rotated_matrix>=0.5)] = 1
    return rotated_matrix
def update_weight(pretrained_dict,updated_model_dict):
    for i in pretrained_dict:
        print(i)
        if i.startswith('net.0.'):
            # print('old name',i)
            tmp = 'feature_net.' + i[len('net.0.'):]
            print('old name',i,'new name',tmp)
            if tmp in updated_model_dict:
                print('update '+tmp)
                updated_model_dict[tmp].copy_(pretrained_dict[tmp])
        # elif i.startswith('net.1.'):
        #     # print('old name',i)
        #     tmp = 'process_net.' + i[len('net.1.'):]
        #     print('old name',i,'new name',tmp)
        #     if tmp in updated_model_dict:
        #         print('update '+tmp)
        #         updated_model_dict[tmp].copy_(pretrained_dict[tmp])
    return updated_model_dict

class Push_Agent():
    def __init__(self,env,num_envs,device,save_path) -> None:
        self.env = env
        self.obs_shape = [2,72,72]
        self.num_envs = num_envs
        learning_rate=0.0001
        mem_size=1000
        self.eps_start=1
        self.eps_end=0.1
        self.eps_decay=120000
        self.save_path = save_path
        depth_only=False
        load_path=None
        self.training=True
        seed=20
        optimizer="ADAM"
        self.device = device
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.Orient = 8
        self.WIDTH = 72
        self.HEIGHT = 72
        self.BATCH_SIZE = 16
        self.GAMMA = 0.75
        self.policy_net = MULTIDISCRETE_RESNET_Rotate(number_actions_dim_2=1)
        # Only need a target network if gamma is not zero
        self.target_net = MULTIDISCRETE_RESNET_Rotate(number_actions_dim_2=1)
        # pretrained_dict = torch.load('/home/chenxiny/orbit/Orbit/FCN_regression/weight31200.pth')
        self.saved_changed = {}
        for i in range(self.num_envs):
            self.saved_changed[i] = []
        # print(self.saved_changed)
        # checkpoint = torch.load('/home/cxy/Downloads/Mar28_mask_weight128000.pth')
        # model_dict = self.policy_net.state_dict()
        ''' TODO: load pretrained
        # updated_model_dict = self.policy_net.state_dict()
        # # print(pretrained_dict.items())
        # # print(updated_model_dict.items())
        # pretrained_dict=update_weight(pretrained_dict,updated_model_dict)
        # # Filter out the pretrained weights for the unchanged layers
        # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in updated_model_dict}

        # # Update the state_dict of the updated model with the pretrained weights for unchanged layers
        # updated_model_dict.update(pretrained_dict)
        # ''' 
        # updated_model_dict = self.policy_net.state_dict()
        # pretrained_dict=update_weight(pretrained_dict,updated_model_dict)
        # self.policy_net.load_state_dict(updated_model_dict)
        # self.target_net.load_state_dict(updated_model_dict)
        # self.policy_net.load_state_dict(checkpoint)
        # self.target_net.load_state_dict(checkpoint)
        # print('load weight',checkpoint)
        self.target_net.eval()
        # print('policy')
        # print(self.policy_net)
        # if load_path is not None:
        #     checkpoint = torch.load(load_path)
        #     self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        #     self.target_net.load_state_dict(checkpoint["model_state_dict"])
        #     print("Successfully loaded weights from {}.".format(load_path))
        if self.training:
            self.memory = ReplayBuffer(mem_size)
            if optimizer == "ADAM":
                self.optimizer = optim.Adam(
                    self.policy_net.parameters(), lr=learning_rate, weight_decay=0.00002
                )
            if load_path is None:
                self.steps_done = 0
                self.eps_threshold = self.eps_start
                date = "_".join(
                    [
                        str(time.localtime()[1]),
                        str(time.localtime()[2]),
                        str(time.localtime()[0]),
                        str(time.localtime()[3]),
                        str(time.localtime()[4]),
                    ]
                )
                '''
                ALGORITHM = "DQN"
                OPTIMIZER = "ADAM"
                '''
                self.DESCRIPTION = "logs"
                
                self.WEIGHT_PATH =  "FCN_" + date + "_weights.pt"
                self.greedy_rotations = defaultdict(int)
                self.greedy_rotations_successes = defaultdict(int)
                self.random_rotations_successes = defaultdict(int)
            # Tensorboard setup
            self.writer = SummaryWriter('logs/FCN/Feb_6_after')
            
            self.last_1000_rewards = deque(maxlen=1000)
            self.last_100_loss = deque(maxlen=100)
            self.last_1000_actions = deque(maxlen=1000)
    def epsilon_greedy(self, state,mask_o,env_idx):
        """
        Returns an action according to the epsilon-greedy policy.

        Args:
            state: An observation / state that will be forwarded through the policy net if greedy action is chosen.
        """

        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        # self.eps_threshold = EPS_STEADY
        self.writer.add_scalar("Epsilon", self.eps_threshold, global_step=self.steps_done)
        self.steps_done += 1
        # if self.steps_done < 2*BATCH_SIZE:
        # self.last_action = 'random'
        # return torch.tensor([[random.randrange(self.output)]], dtype=torch.long)
        if sample > self.eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                # For RESNET
                print('greedy')
                final_max_value = -10
                final_max_idx = 0
                final_mask_tensor = 0
                value_list =[]
                for i in range(12):
                    mask_tmp = mask_o.copy()
                    mask_tmp = rotate_45(mask_tmp,-i*30)
                    mask_tmp = mask_tmp[7:43,7:43]
                    plt.imshow(mask_tmp)
                    plt.show()
                    mask_tmp = mask_tmp[:, :, np.newaxis]
                    mask_tmp = np.moveaxis(mask_tmp, -1, 0)
                    mask_tensor = self.transform_mask(mask_tmp)
                    output = self.policy_net(state.to(self.device),mask_tensor.to(self.device))
                    for idx in self.saved_changed[env_idx]:
                        output[:,idx] = torch.min(output)
                    max_value = output.view(-1).max(0)[0]
                    max_idx = output.view(-1).max(0)[1]
                    output_np = output.cpu().detach().numpy()
                    max_idx = max_idx.view(1)
                    # print('max value',max_value,'idx',max_idx)
                    # print(int(max_idx//(self.WIDTH*self.WIDTH)),int(max_idx%(self.WIDTH*self.WIDTH)//self.WIDTH),int(max_idx%(self.WIDTH*self.WIDTH)%self.WIDTH))
                    if float(max_value.cpu())>final_max_value:
                        final_max_value = float(max_value.cpu())
                        final_max_idx = max_idx.clone()
                        final_mask_tensor = mask_tensor.clone()
                    # value_list.append(float(max_value.cpu()))
                    # value_list.append(int(max_idx.clone().cpu()))
                    if i == 0:
                        print(max_value,max_idx)
                self.saved_changed[env_idx].append(int(final_max_idx//(self.WIDTH*self.WIDTH)))
                # Do not want to store replay buffer in GPU memory, so put action tensor to cpu.
                # print(state.size())
                # image = np.zeros((144,144,3))
                # occu = np.squeeze(state.cpu().numpy())[0]
                # image[:,:,0] = occu.copy()
                # image[:,:,1] = output_np.copy()
                # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 10))
                # occu[int(max_idx/144)-1:int(max_idx/144)+2,int(max_idx%144)-1:int(max_idx%144)+1] = 3
                
                # ax1.imshow(occu)
                # ax2.imshow(output_np)
                # ax3.imshow(image)
                # plt.show()
                # print('greedy')
                # print('####################################')
                # print(output.size())
                # print(max_idx)
                # print('value list',value_list)
                print('final value',final_max_value,'fianl action:',int(final_max_idx//(self.WIDTH*self.WIDTH)),int(final_max_idx%(self.WIDTH*self.WIDTH)//self.WIDTH),int(final_max_idx%(self.WIDTH*self.WIDTH)%self.WIDTH))
                # print(max_value,output_np[int(max_idx/self.WIDTH),int(max_idx%self.WIDTH)])
                # print('####################################')

                # print('predict action',max_idx)
                # print('final mask tensor')
                # print(final_mask_tensor.size())
                # plt.imshow(final_mask_tensor[0,0].cpu().numpy())
                # plt.show()
                return final_max_idx.unsqueeze_(0).cpu(),final_mask_tensor
        # else:
        #     self.last_action = 'random'
        #     return torch.tensor([[random.randrange(self.output)]], dtype=torch.long)

        # Little trick for faster training: When sampling a random action, check the depth value
        # of the selected pixel and resample until you get a pixel corresponding to a point on the table
        else:
            self.last_action = "random"
            action = random.randrange(int(self.Orient*self.WIDTH*self.WIDTH))
            mask_ori = random.randrange(12)
            mask_tmp = mask_o.copy()
            mask_tmp = rotate_45(mask_tmp,-mask_ori*30)
            mask_tmp = mask_tmp[7:43,7:43]
            # plt.imshow(mask_tmp)
            # plt.show()
            mask_tmp = mask_tmp[:, :, np.newaxis]
            mask_tmp = np.moveaxis(mask_tmp, -1, 0)
            mask_tensor = self.transform_mask(mask_tmp)
            return torch.tensor([[action]], dtype=torch.long),mask_tensor

    def greedy(self, state):
        """
        Always returns the greedy action. For demonstrating learned behaviour.

        Args:
            state: An observation / state that will be forwarded through the policy network to receive the action with the highest Q value.
        """

        self.last_action = "greedy"

        with torch.no_grad():
            max_o = self.policy_net(state.to(self.device)).view(-1).max(0)
            max_idx = max_o[1]
            max_value = max_o[0]

            return max_idx, max_value.item()
    def transform_observation(self, observation):
        obs = observation.copy().astype(np.float32)
        obs = obs/float(255)
        obs_tensor = torch.tensor(obs).float()
        # Add batch dimension.
        obs_tensor.unsqueeze_(0)

        return obs_tensor
    def transform_mask(self,mask):
        mask_t = mask.copy().astype(np.float32)
        mask_tensor = torch.tensor(mask_t).float()
        mask_tensor.unsqueeze_(0)
        return mask_tensor
    def transform_action(self, action):
        action_value = action.item()
        action_1 = action_value // (self.WIDTH*self.HEIGHT)
        action_2 = (action_value % (self.WIDTH*self.HEIGHT))//self.HEIGHT
        action_3 = (action_value % (self.WIDTH*self.HEIGHT))%self.HEIGHT
        return np.array([action_2, action_3,action_1])
    def train(self):
        self.save_i = self.steps_done//6000 +1
        self._last_obs = self.env.reset()
        # print('obs shape',self._last_obs.shape)
        i_range = int(np.ceil(self.BATCH_SIZE/len(self._last_obs)))
        while self.steps_done<300000:
            #print("i_range: ",i_range)
            for i in range(i_range):
                actions = np.zeros((len(self._last_obs),3))
                obs_tensor_list = []
                action_list = []
                mask_list = []
                for j in range(len(self._last_obs)):
                    obs_tmp = self._last_obs[j].copy()
                    obs_tmp = np.moveaxis(obs_tmp, -1, 0)
                    mask_tmp = self.env.obj_masks[j]
                    mask_o = mask_tmp.clone().cpu().numpy()
                    
                        # plt.imshow(mask_t)
                        # plt.show()
                    
                    # for k in range(1,12):
                    #     mask_t = rotate_45(mask_o,degree=30*k)
                    #     mask_t = mask_t[7:43,7:43]
                    #     mask_t = mask_t[:, :, np.newaxis]
                    #     mask_t = np.moveaxis(mask_t, -1, 0)
                    #     mask_tmp = np.concatenate((mask_tmp,mask_t),axis=0)
                    # for k in range(12):

                    #     plt.imshow(mask_tmp[k])
                    #     plt.show()
                    # print('obs shape',obs_tmp.shape)
                    obs_tensor = self.transform_observation(obs_tmp)
                    
                    action,mask_tensor = self.epsilon_greedy(obs_tensor,mask_o,j)
                    
                    # print('mask size',mask_tensor.size())
                    # plt.imshow(self.env.obj_masks[j])
                    # plt.show()
                    # print("action: ",action)
                    env_action = self.transform_action(action)
                    #print(int(i*len(self._last_obs)+j),i,j,len(self._last_obs))
                    actions[j,:3] = env_action.flatten()
                    actions[j,2] = actions[j,2]*int(np.ceil(8.0/self.Orient))
                    # print('action:',actions)
                    obs_tensor_list.append(obs_tensor)
                    action_list.append(action)
                    mask_list.append(mask_tensor)
                new_obs, rewards, dones, infos = self.env.step(actions)
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                    ):
                        # print('terminate')
                        next_obs_tmp = infos[idx]["terminal_observation"]
                    else:
                        next_obs_tmp = new_obs[idx].copy()

                    # next_obs_tmp = self.env.tabletop_scene_tmp[idx].copy()
                    if done:
                        self.saved_changed[idx] = []
                    else:
                        if self.env.check_reaching[idx]>0.5:
                            self.saved_changed[idx] = []
                        # print('not terminate')
                    # if done:
                    #     print('terminate')
                    #     if infos[idx].get("terminal_observation") is not None:
                    #         print('donot store the terminate state')
                # for j in range(len(self._last_obs)):
                    # print('create buffer',idx)
                    next_obs_tmp = np.moveaxis(next_obs_tmp, -1, 0)
                    next_obs_tensor = self.transform_observation(next_obs_tmp)
                    # if rewards.flatten()[j]>0.01:
                    #     rewards_tmp = torch.tensor([[1]])
                    # else:
                    #     rewards_tmp = torch.tensor([[0]])
                    rewards_tmp = torch.tensor([[rewards.flatten()[idx]]])*10
                    obs_tmp = obs_tensor_list[idx]
                    action_tmp = action_list[idx]
                    mask_tmp = mask_list[idx]
                    # mask_tmp = mask_tensor_list[idx]
                    # plt.imshow(mask_tmp[0,0].cpu().numpy())
                    # plt.show()
                    self.memory.push(obs_tmp,action_tmp,next_obs_tensor,rewards_tmp,mask_tmp)
                    # print('check_memory')
                    # print('reward',rewards_tmp)
                    # print('action',action_tmp)
                    # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(7, 4))
                    # ax1.imshow(np.array(obs_tmp[0,0,:,:].clone().cpu()))
                    # ax2.imshow(np.array(next_obs_tensor[0,0,:,:].clone().detach().cpu()))
                    
                    # plt.show()
                    del next_obs_tensor,rewards_tmp,action_tmp,obs_tmp
                del action_list,obs_tensor_list
                print(self.steps_done//self.num_envs,self.saved_changed)
                self._last_obs = new_obs.copy()
                self.writer.add_scalar("mean rewards", np.mean(rewards.flatten()), global_step=self.steps_done)
                print("mean rewards: ", np.mean(rewards.flatten()*10))
            self.learn()
            if self.steps_done>=3200*self.save_i:
                self.save_i += 1
                print('num model to be saved: ',self.save_i)
                torch.save(self.policy_net.state_dict(),self.save_path+str(self.steps_done)+'.pth')
                # torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
                print("Saved checkpoint to {}.".format(self.WEIGHT_PATH))
        print(f"Finished training")
        self.writer.close()



                    


    def learn(self):
        """
        Example implementaion of a training method, using standard DQN-learning.
        Samples batches from the replay buffer, feeds them through the policy net, calculates loss,
        and calls the optimizer.
        """

        # Make sure we have collected enough data for at least one batch
        if len(self.memory) < 2 * self.BATCH_SIZE:
            print("Filling the replay buffer ...")
            return

        # Sample the replay buffer
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch for easier access (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*transitions))

        # Gradient accumulation to bypass GPU memory restrictions
        for i in range(1):
            # Transfer weights every TARGET_NETWORK_UPDATE steps
            if self.steps_done % 160 == 0:
                if i == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print('load the weight of policy net to target')
            start_idx = i * (self.BATCH_SIZE)
            end_idx = (i + 1) * (self.BATCH_SIZE)

            state_batch = torch.cat(batch.state[start_idx:end_idx]).to(self.device)
            action_batch = torch.cat(batch.action[start_idx:end_idx]).to(self.device)
            next_state_batch = torch.cat(batch.next_state[start_idx:end_idx]).to(self.device)
            reward_batch = torch.cat(batch.reward[start_idx:end_idx]).to(self.device)
            mask_batch = torch.cat(batch.mask[start_idx:end_idx]).to(self.device)
            # Current Q prediction of our policy net, for the actions we took
            q_pred = (
                self.policy_net(state_batch,mask_batch).view(self.BATCH_SIZE, -1).gather(1, action_batch)
            )
            # print('q_pred')
            # print(self.policy_net(state_batch).size())
            # q_pred = self.policy_net(state_batch).gather(1, action_batch)
            # print('q_pred',q_pred)

            q_next_state = self.target_net(next_state_batch,mask_batch).view(self.BATCH_SIZE, -1).max(1)[0].unsqueeze(1).detach()
            # print('q_pred')
            # print(self.target_net(next_state_batch).size())
                # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
            q_expected = reward_batch + (self.GAMMA * q_next_state)
            # print('q_expected',q_expected)
            criterien=nn.SmoothL1Loss()
            # q_expected = reward_batch.float()
            # loss = F.binary_cross_entropy(q_pred, q_expected,reduction='mean')
            loss = criterien(q_pred, q_expected)
            loss.backward()
            print('backpropagate',i)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.last_100_loss.append(loss.item())
            self.writer.add_scalar('Average loss', loss, global_step=self.steps_done)
            print('loss: ',loss,'steps',self.steps_done)
        
