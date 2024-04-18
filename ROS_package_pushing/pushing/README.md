
# Preparations

## Install additional python libraries:
```
pip install stable-baselines3
pip install opencv-python
pip install open3d
pip install shapely
pip install scipy
```

## Download the PPO with CNN pushing model

please down load the [trained model](https://drive.google.com/drive/folders/1Cs4M6IC1g8I4HtM5DW9w-0GQS64BWv6l?usp=sharing), and save the model under the 'data' directory.


## Table calibration

Please calibrate the table following the [instructions](https://github.com/ethz-asl/moma_docs/wiki/Panda-Software-Quickstart#step-1-simulation), before applying the next step.

# Apply pushing methods

## start the controllers

run with moveit:

```
roslaunch moma_bringup panda_real.launch moveit:=true
```

run the sensor:

```
roslaunch moma_bringup sensors.launch wrist_camera:=true fixed_camera:=false
```

## apply pushing and placing methods:

run proposed placing method without pushing:
```
rosun pushing placing.py
```

run placing baseline without pushing:
```
rosun pushing placing_baseline.py
```

run the pushing baseline:
```
rosrun pushing pushing_compare.py
```

run the PPO with CNN pushing method:
```
rosrun pushing pushing_new.py model_path:= relative file path to the model, e.g.: ./data/PPO_model.zip
```