# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""


import argparse
import numpy as np
import torch
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper

from config import parse_sb3_cfg


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)
    # parse agent configuration
    agent_cfg = parse_sb3_cfg(args_cli.task)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # check checkpoint is valid
    if args_cli.checkpoint is None:
        raise ValueError("Checkpoint path is not valid.")
    # create agent from stable baselines
    print(f"Loading checkpoint from: {args_cli.checkpoint}")
    agent = PPO.load(args_cli.checkpoint, env, print_system_info=True)
    # torch.save(agent.policy.state_dict(),'/home/cxy/Downloads/436800weight.pth')
    # reset environment
    obs = env.reset()
    stop_pushing = 0
    # simulate environment
    print('using stop pushing method')
    while simulation_app.is_running():
        # agent stepping
        act_app = np.zeros(len(obs))
        actions, _ = agent.predict(obs, deterministic=True)
        obs_tensor = torch.from_numpy(obs).cuda()
        # print(obs_tensor.size())
        obs_tensor = obs_tensor.permute(0,3,1,2)
        # print(obs_tensor.size())
        actions_tensor_tmp =  torch.from_numpy(actions).cuda()
        value,log_prob,entropy = agent.policy.evaluate_actions(obs_tensor,actions_tensor_tmp)
        # print('value log prob entropy')
        # print(value,log_prob,entropy)
        obs_tmp = obs.copy()
        obs_tensor_tmp = obs_tensor.detach().clone()
        for j in range(3):
            obs_tmp = np.rot90(obs_tmp,1,(2,1))
            obs_tmp = obs_tmp.copy()
            obs_tensor_tmp = obs_tensor_tmp.rot90(1,[3,2])
            actions_tmp, _ = agent.predict(obs_tmp, deterministic=True)
            actions_tensor_tmp =  torch.from_numpy(actions_tmp).cuda()
            value_tmp,log_prob_tmp,entropy_tmp = agent.policy.evaluate_actions(obs_tensor_tmp,actions_tensor_tmp)
            for i in range(len(obs_tensor)):
                # if float(log_prob_tmp[i])>float(log_prob[i]):
                if float(value_tmp[i]) > float(value[i]):
                    actions[i] = actions_tmp[i]
                    act_app[i] = j * 2.0 +2.0
                    log_prob[i] = log_prob_tmp[i]
                    value[i] = value_tmp[i]
        actions_origin = actions.copy()
        for _ in range(len(obs)):
            if act_app[_] == 2:
                actions[_,0] = 49-actions[_,1]
                actions[_,1] = actions_origin[_,0]
            elif act_app[_] == 4:
                actions[_,0] = 49-actions[_,0]
                actions[_,1] = 49-actions_origin[_,1]
            elif act_app[_] == 6:
                actions[_,0] = actions[_,1]
                actions[_,1] = 49-actions_origin[_,0]
        for _ in range(len(value)):
            if float(value[_]) <=-0.1:
                act_app[_] = 10
        actions_new = np.c_[actions,act_app.T]    
        for _ in env.env.stop_pushing.tolist():
            if _ >= 0.5:
                stop_pushing +=1
                print('stop pushing')
                print(stop_pushing)
        # print(actions_new)
        # print(_)
        # print(obs)
        # print(obs.shape)
        ####################################### add by xy Dec 19
        # value,log_prob,entropy = agent.policy.evaluate_actions()
        #######################################
        # env stepping
        obs, _, _, _ = env.step(actions_new)
        
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
