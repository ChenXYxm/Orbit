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
    # if args_cli.checkpoint is None:
    #     raise ValueError("Checkpoint path is not valid.")
    # create agent from stable baselines
    # print(f"Loading checkpoint from: {args_cli.checkpoint}")
    # agent_cfg = parse_sb3_cfg(args_cli.task)
    # override configuration with command line arguments
    # if args_cli.seed is not None:
    #     agent_cfg["seed"] = args_cli.seed
    # policy_arch = agent_cfg.pop("policy")
    # agent = PPO.load(args_cli.checkpoint, env, print_system_info=True)
    # agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    # state_dict = torch.load('/home/cxy/Downloads/weight758080.pth')
    # agent.policy.load_state_dict(state_dict)
    # torch.save(agent.policy.state_dict(),'/home/cxy/Downloads/436800weight.pth')
    # reset environment
    obs = env.reset()
    stop_pushing = 0
    # simulate environment
    # print('using stop pushing method')
    x_start = 48
    flag_compare = True
    while simulation_app.is_running():
        # agent stepping
        act_app = np.zeros(len(obs))
        actions = np.zeros((len(obs),2))
        value = np.zeros(len(obs))
        for _ in range(len(actions)):
            if not flag_compare:
                if float(value[_]) <=-0.12:
                    act_app[_] = 10
        ################# TODO:only for comparision method
            if flag_compare:
                act_app[_] = 0
                actions[_,0] = x_start
                actions[_,1] = 49
        #######################################
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
        # print(actions_new)
        obs, _, dones, _ = env.step(actions_new)

        ################# TODO:only for comparision method
        if flag_compare:
            x_start = x_start -3
            if x_start <2:
                x_start = 49
            for idx, done in enumerate(dones):
                if done:
                    x_start = 49
        ############################
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
