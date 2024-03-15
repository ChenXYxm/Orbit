import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Modules import ReplayBuffer, Transition, simple_Transition,MULTIDISCRETE_RESNET
import numpy as np
import pickle
import random
import copy
import math
from collections import deque, defaultdict
import time
import matplotlib.pyplot as plt
class Push_Agent():
    def __init__(self,env,num_envs,device) -> None:
        torch.cuda.empty_cache()
        self.env = env
        self.obs_shape = [2,64,64]
        self.num_envs = num_envs
        learning_rate=0.001
        mem_size=300
        self.eps_start=1
        self.eps_end=0.2
        self.eps_decay=10000
        depth_only=False
        load_path=None
        self.training=True
        seed=20
        optimizer="ADAM"
        self.device = device
        # torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.WIDTH = 64
        self.HEIGHT = 64
        self.BATCH_SIZE = 3
        self.GAMMA = 0.9
        self.policy_net = MULTIDISCRETE_RESNET(number_actions_dim_2=1).to(self.device)
        # Only need a target network if gamma is not zero
        self.target_net = MULTIDISCRETE_RESNET(number_actions_dim_2=1).to(self.device)
        # checkpoint = torch.load('/home/cxy/Downloads/FCN_env/weight234000.pth')
        checkpoint = torch.load('/home/chenxiny/orbit/Orbit/FCN_regression/weight33000.pth')
        # print('structure')
        # print(self.policy_net)
        self.policy_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(checkpoint)
        self.target_net.eval()
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
    def epsilon_greedy(self, state,mask):
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
                
                output = self.policy_net(state.to(self.device),mask.to(self.device))
                max_value = output.view(-1).max(0)[0]
                max_idx = output.view(-1).max(0)[1]
                output_np = output.cpu().detach().numpy()
                max_idx = max_idx.view(1)
                # Do not want to store replay buffer in GPU memory, so put action tensor to cpu.
                # print(state.size())
                # max_1 = max_idx//int(144*144)
                # max_idx_2 = max_idx%int(144*144)
                # image = np.zeros((144,144,3))
                # occu = np.squeeze(state.cpu().numpy())[0]
                # image[:,:,0] = occu.copy()
                # image[:,:,1] = output_np[0,max_1].copy()
                # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 10))
                # occu[int(max_idx_2/144)-4:int(max_idx_2/144)+5,int(max_idx_2%144)-3:int(max_idx_2%144)+4] = 3
                
                # ax1.imshow(occu)
                # ax2.imshow(output_np[0,max_1])
                # ax3.imshow(image)
                # plt.show()
                # print('action:',np.unravel_index(np.argmax(output_np),output_np.shape))
                # print('greedy')
                # print('####################################')
                # print(output.size())
                # print(max_idx)
                # print(int(max_idx/144),int(max_idx%144))
                # print(max_value,output_np[int(max_idx/144),int(max_idx%144)])
                # print('####################################')

                # print('predict action',max_idx)
                del output, max_value,output_np
                return max_idx.unsqueeze_(0).cpu()
        # else:
        #     self.last_action = 'random'
        #     return torch.tensor([[random.randrange(self.output)]], dtype=torch.long)

        # Little trick for faster training: When sampling a random action, check the depth value
        # of the selected pixel and resample until you get a pixel corresponding to a point on the table
        else:
            self.last_action = "random"
            action = random.randrange(int(64*64*1))
                

            return torch.tensor([[action]], dtype=torch.long)

    def greedy(self, state,mask):
        """
        Always returns the greedy action. For demonstrating learned behaviour.

        Args:
            state: An observation / state that will be forwarded through the policy network to receive the action with the highest Q value.
        """

        self.last_action = "greedy"

        with torch.no_grad():
            max_o = self.policy_net(state.to(self.device),mask.to(self.device)).view(-1).max(0)
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
    def transform_action(self, action):
        action_value = action.item()
        action_1 = action_value // int(64*64)
        action_2 = (action_value % int(64*64))//64
        action_3 = (action_value % int(64*64))%64
        return np.array([action_2, action_3,action_1])
    def train(self):
        self.save_i = 1
        self._last_obs = self.env.reset()
        # print('obs shape',self._last_obs.shape)
        i_range = int(np.ceil(3/len(self._last_obs)))
        while self.steps_done<300000:
            #print("i_range: ",i_range)
            for i in range(i_range):
                actions = np.zeros((len(self._last_obs),3))
                obs_tensor_list = []
                action_list = []
                mask_tensor_list = []
                for j in range(len(self._last_obs)):
                    obs_tmp = self._last_obs[j].copy()
                    obs_tmp = np.moveaxis(obs_tmp, -1, 0)
                    mask = self.env.new_obj_mask[j]
                    
                    # plt.imshow(mask)
                    # plt.show()
                    # print('obs train shape',obs_tmp.shape)
                    obs_tensor = self.transform_observation(obs_tmp)
                    # print('obs shape shape',obs_tensor.size())
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze_(dim=0)
                    mask_tensor = mask_tensor.unsqueeze_(dim=0)
                    # print("mask shape: ", mask_tensor.size())
                    action = self.epsilon_greedy(obs_tensor,mask_tensor)
                    # print("action: ",action)
                    env_action = self.transform_action(action)
                    #print(int(i*len(self._last_obs)+j),i,j,len(self._last_obs))
                    actions[j,:3] = env_action.flatten()
                    # actions[j,2] = 3
                    # actions[j,1] = 24
                    # actions[j,0] = 30
                    obs_tensor_list.append(obs_tensor)
                    action_list.append(action)
                    mask_tensor_list.append(mask_tensor)
                    del obs_tensor,mask_tensor,action
                new_obs, rewards, dones, infos = self.env.step(actions)
                for j in range(len(self._last_obs)):
                    next_obs_tmp = new_obs[j].copy()
                    next_obs_tmp = np.moveaxis(next_obs_tmp, -1, 0)
                    next_obs_tensor = self.transform_observation(next_obs_tmp)
                    # if rewards.flatten()[j]>0.01:
                    #     rewards_tmp = torch.tensor([[1]])
                    # else:
                    #     rewards_tmp = torch.tensor([[0]])
                    rewards_tmp = torch.tensor([[rewards.flatten()[j]]])*10
                    obs_tmp = obs_tensor_list[j]
                    action_tmp = action_list[j]
                    mask_tmp = mask_tensor_list[j]
                    self.memory.push(obs_tmp,mask_tmp,action_tmp,next_obs_tensor,rewards_tmp)
                    del next_obs_tensor,rewards_tmp,action_tmp,mask_tmp,obs_tmp
                self._last_obs = new_obs.copy()
                self.writer.add_scalar("mean rewards", np.mean(rewards.flatten()), global_step=self.steps_done)
                print("mean rewards: ", np.mean(rewards.flatten()*10))
            self.learn()
            if self.steps_done>=3000*self.save_i:
                self.save_i +=1
                
                torch.save(self.policy_net.state_dict(),'FCN_regression/weight'+str(self.steps_done)+'.pth')
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
            if self.steps_done % 64 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print('load the weight of policy net to target')
            start_idx = i *3
            end_idx = (i + 1) * 3

            state_batch = torch.cat(batch.state[start_idx:end_idx]).to(self.device)
            action_batch = torch.cat(batch.action[start_idx:end_idx]).to(self.device)
            next_state_batch = torch.cat(batch.next_state[start_idx:end_idx]).to(self.device)
            reward_batch = torch.cat(batch.reward[start_idx:end_idx]).to(self.device)
            mask_batch = torch.cat(batch.mask[start_idx:end_idx]).to(self.device)
            # Current Q prediction of our policy net, for the actions we took
            # print('batch state size', state_batch.size())
            # print('batch mask size', mask_batch.size())
            # print('batch next state size', next_state_batch.size())
            # print('batch action size', action_batch.size())
            # print('reward size', reward_batch.size())
            q_pred = (
                self.policy_net(state_batch,mask_batch).view(3, -1).gather(1, action_batch)
            )
            # print('q_pred')
            # print(self.policy_net(state_batch).size())
            # q_pred = self.policy_net(state_batch).gather(1, action_batch)
            # print('q_pred',q_pred)

            q_next_state = self.target_net(next_state_batch,mask_batch).view(3, -1).max(1)[0].unsqueeze(1).detach()
            print('q_pred')
            # print(self.target_net(next_state_batch).size())
                # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
            q_expected = reward_batch + (self.GAMMA * q_next_state)
            # print('q_expected',q_expected)
            criterien=nn.SmoothL1Loss()
            # q_expected = reward_batch.float()
            # loss = F.binary_cross_entropy(q_pred, q_expected,reduction='mean')
            loss = criterien(q_pred, q_expected)
            loss.backward()
            del state_batch,action_batch,next_state_batch,reward_batch, q_pred, q_next_state, q_expected
        self.last_100_loss.append(loss.item())
        self.writer.add_scalar('Average loss', loss, global_step=self.steps_done)
        print('loss: ',loss,'steps',self.steps_done)
        self.optimizer.step()

        self.optimizer.zero_grad()
