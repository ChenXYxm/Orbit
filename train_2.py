"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

# create isaac environment
import gym

import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import load_default_env_cfg
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg
# create base environment
# cfg = load_default_env_cfg("Isaac-Lift-Franka-v0")
cfg = parse_env_cfg("Isaac-Lift-Franka-v0",num_envs=10)
# cfg["env"]["num_envs"] = 10
env = gym.make("Isaac-Lift-Franka-v0", cfg=cfg, headless=True)

from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper
# https://isaac-orbit.github.io/orbit/source/setup/installation.html
env = Sb3VecEnvWrapper(env)

# import stable baselines
from stable_baselines3 import SAC

# create agent from stable baselines
model = SAC(
    "MlpPolicy",
    env,
    batch_size=1000,
    learning_rate=0.001,
    gamma=0.99,
    device="cuda:0",
    ent_coef='auto',
    verbose=1,
    tensorboard_log="./cartpole_tensorboard",
)
model.learn(total_timesteps=1000000)
model.save("sac_cartpole_sb3paral")

env.close()