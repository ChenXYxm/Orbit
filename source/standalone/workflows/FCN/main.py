# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""


import argparse
import numpy as np
import os

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""


import gym
import os
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from trainer import Trainer
from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    

    # directory for logging into
    # logs/sb3/Isaac-Toy-Franka-v0/Dec06_17-42-42
    # log_dir = os.path.join("logs", "sb3", 'Isaac-Toy-Franka-v0', 'Dec06_17-42-42')
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%b%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video)
    # wrap for video recording
    env = Sb3VecEnvWrapper(env)
    # set the seed
    env.seed(seed=36)
    writer = SummaryWriter(log_dir=log_dir)
    obs = env.reset()
    ################################# cfg
    future_reward_discount = 0.5
    use_cude = True
    trainer = Trainer(future_reward_discount=future_reward_discount)
    explore_prob = 0.1
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'reward' : 0.0,}
    run_steps=0
    def process_actions():
        best_push_conf = np.max(push_predictions)
        best_stop_conf = np.max(stop_predictions)
        if best_push_conf > best_stop_conf:
            nonlocal_variables['primitive_action'] = 'push'
        else:
            nonlocal_variables['primitive_action'] = 'stop'
        explore_actions = np.random.uniform() < explore_prob
        if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                
            nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0,2) == 0 else 'stop'
        if nonlocal_variables['primitive_action'] == 'push':
            nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
            predicted_value = np.max(push_predictions)
        elif nonlocal_variables['primitive_action'] == 'stop':
            nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(stop_predictions), stop_predictions.shape)
            predicted_value = np.max(stop_predictions)
        # trainer.predicted_value_log.append([predicted_value])
        best_rotation_angle = nonlocal_variables['best_pix_ind'][0]
        best_pix_x = nonlocal_variables['best_pix_ind'][2]
        best_pix_y = nonlocal_variables['best_pix_ind'][1]
        
        nonlocal_variables['executing_action'] = False

    while True:
        run_steps +=1
        push_predictions, stop_predictions, state_feat = trainer.forward(obs, is_volatile=True)
        nonlocal_variables['executing_action'] = True
        process_actions()
        best_pix_ind=np.array([nonlocal_variables['best_pix_ind'][2],nonlocal_variables['best_pix_ind'][1],nonlocal_variables['best_pix_ind'][0]])
        best_pix_ind=best_pix_ind.reshape((1,-1))
        if nonlocal_variables['primitive_action'] == 'stop':
            # best_pix_ind[:,:] = 150
            new_obs, rewards, dones, infos = env.step(np.array([[150,150,150]]))
        else:
            new_obs, rewards, dones, infos = env.step(best_pix_ind)
        if nonlocal_variables['primitive_action'] == 'stop':
            rewards = 0.0
        writer.add_scalar(tag='rewards',scalar_value=float(rewards),global_step=run_steps)
        label_value, prev_reward_value = trainer.get_label_value(reward=rewards,primitive_action=nonlocal_variables["primitive_action"],
                                                                 prev_push_predictions=push_predictions,
                                                                 prev_stop_predictions=stop_predictions,
                                                                 next_image=new_obs)
        trainer.backprop(primitive_action=nonlocal_variables["primitive_action"],best_pix_ind=nonlocal_variables["best_pix_ind"],
                         depth_heightmap=obs,label_value=label_value)
        obs = new_obs.copy()
        if run_steps%1000 == 0:
            path_weight = log_dir +'/'+'weight_'+str(run_steps)+'.pth'
            torch.save(trainer.model.state_dict(),path_weight)
        if run_steps>=15000:
            break




    # create agent from stable baselines
    
    # state_dict = torch.load('/home/cxy/Thesis/orbit/Orbit/logs/sb3/Isaac-Toy-Franka-v0/Dec11_07-11-07/weight.pth')
    # agent.policy.load_state_dict(state_dict)
    # print(agent.policy)
    # checkpoint_path = '/home/cxy/Thesis/orbit/Orbit/logs/sb3/Isaac-Toy-Franka-v0/Dec11_07-11-07/model_105600_steps'
    # agent = PPO.load(checkpoint_path, env, print_system_info=True)
    # torch.save(agent.policy.state_dict(),'/home/cxy/Thesis/orbit/Orbit/logs/sb3/Isaac-Toy-Franka-v0/Dec11_07-11-07/weight.pth')
    # print(agent.policy)
    # configure the logger
    

    # close the simulator
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()