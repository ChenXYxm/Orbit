
---

# Pushing Assisted Placing on a Tabletop

Models for this project are trained using Omniverse Isaac Sim 2022.2.1 and Orbit v0.1.0



[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-2022.2.1-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![IssacOrbit](https://img.shields.io/badge/Isaac%20Orbit-v0.1.0-red.svg)](https://isaac-orbit.github.io/orbit/source/setup/installation.html)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-lightgrey.svg)](https://releases.ubuntu.com/20.04/)


<!-- TODO: Replace docs status with workflow badge? Link: https://github.com/isaac-orbit/orbit/actions/workflows/docs.yaml/badge.svg -->

Please refer the [Isaac Orbit documentation page](https://isaac-orbit.github.io/orbit) to learn more about the installation of Isaac Sim and Isaac Orbit.We ran this code on Isaac Sim 2022.2.1. Therefore, to execute this code, please install Isaac Sim 2022.2.1 to ensure there are no compatibility issues.

When istalling Isaac Orbit, please clone the orbit repository here, instead of [official Isaac Orbit repository](https://github.com/NVIDIA-Omniverse/orbit). After setting up the environment following the [link](https://isaac-orbit.github.io/orbit), please install open3d, opencv-python, shapely in the created virtual environment.

The trained models for this project can be downloaded from [trained models](https://drive.google.com/drive/folders/1P5K97kQskJ9YJLv1fqWs48eTtYFPR0Nr?usp=drive_link).

## How to run the code


To train the framework of DQN with FCN method without mask, please run:

```
./orbit.sh -p source/standalone/workflows/FCN_method/FCN_without_mask/train.py --num_envs 1 --task Isaac-Push-50-FCN-val-Franka-v0 --headless --save_path /logs/ # save_path indicates the path of the directory to save the trained weight
```


To visualize the performance of the trained model of DQN with FCN method without mask, please run:

```
./orbit.sh -p source/standalone/workflows/FCN_method/FCN_without_mask/play.py --num_envs 1 --task Isaac-Push-50-FCN-val-Franka-v0 --checkpoint /path # checkpoint indicates the path to the trained model weight
```

To train the framework of DQN with FCN method with mask, please run:

```
./orbit.sh -p source/standalone/workflows/FCN_method/FCN_with_mask/train.py --num_envs 1 --task Isaac-Push-50-FCN-val-Franka-v0 --headless --save_path /logs/ # save_path indicates the path of the directory to save the trained weight
```


To visualize the performance of the trained model of DQN with FCN method with mask, please run:

```
./orbit.sh -p source/standalone/workflows/FCN_method/FCN_with_mask/play.py --num_envs 1 --task Isaac-Push-50-FCN-val-Franka-v0 --checkpoint /path # checkpoint indicates the path to the trained model weight
```
