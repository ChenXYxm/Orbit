# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim with lights and a ground plane."""

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


"""Rest everything follows."""
import open3d as o3d
import numpy as np
import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim,GeometryPrim
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
from omni.isaac.orbit.utils.math import convert_quat, quat_mul, random_yaw_orientation, sample_cylinder
import scipy.spatial.transform as tf
from omni.isaac.franka import Franka, KinematicsSolver
"""
Main
"""


def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch",device='cuda:0')
    # Set main camera
    set_camera_view([0, 2.5, 3.5], [0.0, 0.0, 0.0])
    ###########################################
    ###########################################
    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    ###########################################
    ###########################################
    ##################################### scene
    # table_path = f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd"
    # prim_utils.create_prim("/World/Table", usd_path=table_path,position=(0,0,-0.25),scale=(1,0.6,0.5))
    Table = DynamicCuboid(prim_path="/World/Table",position=(0,0,0.20),scale=(1,0.6,0.15))
    Table.set_mass(1000000000)
    sideTable = DynamicCuboid(prim_path="/World/sideTable",position=(0,-0.8,0.20),scale=(0.4,0.4,0.15))
    sideTable.set_mass(100000000)
    ################################ robot setting
    # robot = Franka(prim_path="/World/Robot",position=[0.0, -.45, 0])
    # robot_controller = KinematicsSolver(robot_articulation=robot)
    robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    robot_cfg.data_info.enable_jacobian = True
    robot_cfg.rigid_props.disable_gravity = True
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/Robot", translation=(0.0, -.45, 0))
    ###################################### controller
    ik_control_cfg = DifferentialInverseKinematicsCfg(
        command_type="pose_abs",
        ik_method="dls",
        position_offset=robot.cfg.ee_info.pos_offset,
        rotation_offset=robot.cfg.ee_info.rot_offset,
    )
    ik_controller = DifferentialInverseKinematics(cfg=ik_control_cfg, num_robots=1, device=sim.device)
    ######################################### load ycb objects
    ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        "mug":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/025_mug.usd",
        "largeMarker":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/040_large_marker.usd",
        "tunaFishCan":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
        "banana":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd",
        "pitcherBase":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
        "bowl":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd",
        "largeClamp":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/051_large_clamp.usd",
        "scissors":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/037_scissors.usd",
    }
    ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
                'banana','pitcherBase','bowl','largeClamp','scissors']
    ############################# camera
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device="cuda")
    hand_camera = Camera(cfg=camera_cfg,device='cuda')
    # hand_camera.spawn("/World/Robot/panda_hand/hand_camera", translation=(0.1, 0.0, 0.0),orientation=(0,0,1,0))
    hand_camera.spawn("/World/hand_camera")
    # Spawn camera
    camera.spawn("/World/CameraSensor")
    #############################
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    for _ in range(30):
        sim.step()
    ###################### initialize
    hand_camera.initialize()
    camera.initialize()
    robot.initialize()
    # ik_controller.initialize()
    # # Reset states
    # robot.reset_buffers()
    # ik_controller.reset_idx()
    ###########################
    position = [0, 0, 2.]
    orientation = [0, 0, -1, 0]
    camera.set_world_pose_ros(position, orientation)
    hand_camera.set_world_pose_ros([0,-0.8,2], orientation)
    ###################### fix table position and orientation
    # Table.disable_rigid_body_physics()
    # sideTable.disable_rigid_body_physics()
    ###################### load ycb
    obj_dict = dict()
    for _ in range(6):
        randi = np.random.randint(0,len(ycb_name))
        angle = np.random.randint(0,180)
        key_ori = ycb_name[randi]
        usd_path = ycb_usd_paths[key_ori]
        if key_ori not in obj_dict:
            obj_dict[key_ori] = 1
        else:
            obj_dict[key_ori] +=1
        key = key_ori+str(obj_dict[key_ori])
    # for key, usd_path in ycb_usd_paths.items():
        translation = torch.rand(3).tolist()
        translation = [translation[0]*0.8-0.4,0.45*translation[1]-0.225,0.3]
        # translation = [translation[0]*0.8-0.4,0.8*translation[1]+0.5,0.1]
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
        if key_ori in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl"]:
           rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,angle,0), degrees=True).as_quat(), to="wxyz")
        prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
        GeometryPrim(f"/World/Objects/{key}",collision=True)
        RigidPrim(f"/World/Objects/{key}",mass=0.5)
    for _ in range(50):
        sim.step()
    # Simulate physics
    get_pcd(camera)
    ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)
    ee_goals = [0.0, 0.5, 0.5, 0.0, 0, 0.0,0]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    ik_commands[:] = ee_goals
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            print("pause")
            continue
        ########################################## ik control
        # perform step
        # set the controller commands
        ik_controller.set_command(ik_commands)
        # compute the joint commands
        robot_actions[:, : robot.arm_num_dof] = ik_controller.compute(
            robot.data.ee_state_w[:, 0:3],
            robot.data.ee_state_w[:, 3:7],
            robot.data.ee_jacobian,
            robot.data.arm_dof_pos,
        )
        # in some cases the zero action correspond to offset in actuators
        # so we need to subtract these over here so that they can be added later on
        arm_command_offset = robot.data.actuator_pos_offset[:, : robot.arm_num_dof]
        # offset actuator command with position offsets
        # note: valid only when doing position control of the robot
        robot_actions[:, : robot.arm_num_dof] -= arm_command_offset
        # apply actions
        print('robot action')
        print(robot_actions)
        robot.apply_action(robot_actions)
        # perform step
        # perform step
        print(sim.device)
        sim.step()
def get_pcd(camera):
    camera.update(dt=0.0)
    pointcloud_w, pointcloud_rgb = create_pointcloud_from_rgbd(
            camera.data.intrinsic_matrix,
            depth=camera.data.output["distance_to_image_plane"],
            rgb=camera.data.output["rgb"],
            position=camera.data.position,
            orientation=camera.data.orientation,
            normalize_rgb=True,
            num_channels=4,
    )
    if not isinstance(pointcloud_w, np.ndarray):
        pointcloud_w = pointcloud_w.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_w)
    o3d.visualization.draw_geometries([pcd])
def random_new_object(ycb_usd_paths,ycb_name,obj_dict):
    randi = np.random.randint(0,len(ycb_name))
    key_ori = ycb_name[randi]
    usd_path = ycb_usd_paths[key_ori]
    if key_ori not in obj_dict:
        obj_dict[key_ori] = 1
    else:
        obj_dict[key_ori] +=1
    key = key_ori+str(obj_dict[key_ori])
# for key, usd_path in ycb_usd_paths.items():
    translation = torch.rand(3).tolist()
    translation = [0,-0.8,0.15]
    # translation = [translation[0]*0.8-0.4,0.8*translation[1]+0.5,0.1]
    rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,translation[0]*90), degrees=True).as_quat(), to="wxyz")
    if key_ori in ["mug","tomatoSoupCan","pitcherBase"]:
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,0,translation[0]*90), degrees=True).as_quat(), to="wxyz")
    elif key_ori in ["bowl"]:
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (-180,0,translation[0]*90), degrees=True).as_quat(), to="wxyz")
    prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
    GeometryPrim(f"/World/Objects/{key}",collision=True)
    RigidPrim(f"/World/Objects/{key}",mass=0.5)
    
if __name__ == "__main__":
    # Run empty stage
    main()
    # Close the simulator
    simulation_app.close()