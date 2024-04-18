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