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
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""
import os
import torch
import cv2
import numpy as np
import open3d as o3d
from pxr import Gf, UsdGeom
import random

import omni.isaac.debug_draw._debug_draw as omni_debug_draw
from omni.isaac.core.prims import RigidPrim


from pxr import Gf, UsdGeom

from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.cloner import GridCloner
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
import omni.replicator.core as rep
from omni.isaac.core.utils import prims
from PIL import Image
import matplotlib.pyplot as plt
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
import asyncio
"""
Main
"""


def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    # sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01)
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([0, 1, 3.], [0.0, 0.0, 0])
    # Enable GPU pipeline and flatcache
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)


    # Create interface to clone the scene
    # cloner = GridCloner(spacing=2.0)
    # cloner.define_base_env("/World/envs")
    # # Everything under the namespace "/World/envs/env_0" will be cloned
    # prim_utils.define_prim("/World/envs/env_0")
    # ee_marker = StaticMarker("/Visuals/ee_current", count=1, scale=(0.1, 0.1, 0.1))
    # goal_marker = StaticMarker("/Visuals/ee_goal", count=1, scale=(0.1, 0.1, 0.1))
    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane",z_position=-0.5,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,)
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
    #################### create table 
    table_path = f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd"
    prim_utils.create_prim("/World/Table", usd_path=table_path,position=(0,0,-0.25),scale=(1,0.6,0.5))
    prim_utils.create_prim("/World/Robotbase", usd_path=table_path,position=(0,-0.45,-0.2),scale=(0.3,0.26,0.4))
    # rigid_obj = RigidPrim(f"/World/Table", mass=5.0)
    # # cast to geom prim
    # geom_prim = getattr(UsdGeom, "Cube")(rigid_obj.prim)
    # # set random color
    # color = Gf.Vec3f(0.1, 0.1, 0.2)
    # geom_prim.CreateDisplayColorAttr()
    # geom_prim.GetDisplayColorAttr().Set([color])

    # rigid_obj = RigidPrim(f"/World/Robotbase", mass=5.0)
    # # cast to geom prim
    # geom_prim = getattr(UsdGeom, "Cube")(rigid_obj.prim)
    # # set random color
    # color = Gf.Vec3f(0.1, 0.1, 0.1)
    # geom_prim.CreateDisplayColorAttr()
    # geom_prim.GetDisplayColorAttr().Set([color])
    ################################ robot setting
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

    ###################################### sensor extension camera
    # camera = Camera(

    #     prim_path="/World/camera",

    #     position=np.array([0.0, 0.0, 5.0]),

    #     frequency=20,

    #     resolution=(256, 256),

    #     orientation= rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),

    # )
    # camera.initialize()
    # intrinsics_m = camera.get_intrinsics_matrix() 
    # print(intrinsics_m)
    
    ###################################### rep camera
    
    image_pixel = [512,256]
    cam = rep.create.camera(position=(0,0,5), look_at=(0,0,0))
    rp = rep.create.render_product(cam, image_pixel)
    sim.reset()
    # for _ in range(14):
    #     sim.render()
    
    robot.initialize()
    ik_controller.initialize()
    # Reset states
    robot.reset_buffers()
    ik_controller.reset_idx()

    rgb_image = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_image = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    camera_params = rep.AnnotatorRegistry.get_annotator("CameraParams")
    point_cloud_get = rep.AnnotatorRegistry.get_annotator("pointcloud",init_params={"includeUnlabelled": True}).attach(rp)
    rcp = camera_params.attach(rp)
    rgb_image.attach(rp)
    depth_image.attach(rp)

    rep.orchestrator.step()
    rgb = rgb_image.get_data()
    rep.orchestrator.step()
    depth = depth_image.get_data()
    print(depth)
    rep.orchestrator.step()
    point_cloud = point_cloud_get.get_data()
    rcp_data = rcp.get_data()
    print(point_cloud)
    
    #####################################################################
    # Play the simulator
    
    # Now we are ready!
    
    print("[INFO]: Setup complete...")
    
    points = []
    points_rgb = []
    points.append(point_cloud["data"])
    points_rgb.append(point_cloud["info"]["pointRgb"].reshape(-1, 4)[:, :3])
    pc_data = np.concatenate(points)
    pc_rgb = np.concatenate(points_rgb)
    # print(pc_rgb)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)
    # pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    o3d.visualization.draw_geometries([pcd])
    # print(np.asarray(pcd.colors))
    ply_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
    os.makedirs(ply_out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(ply_out_dir, "pointcloud.ply"), pcd)
    # print(rgb.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    img = Image.fromarray(rgb[:,:,:3])
    ax.scatter(pc_data[:,0], pc_data[:,1], pc_data[:,2])
    plt.show()
    plt.imshow(img)
    plt.show()


    # Create buffers to store actions
    ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)
    if simulation_app.is_running():
        print('running')
    # Set end effector goals
    # Define goals for the arm
    ee_goals = [0, 0, 1, 0.707, 0, 0.707, 0]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    ik_commands[:] = ee_goals
    if simulation_app.is_running():
        print('running')
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    # Note: We need to update buffers before the first step for the controller.
    robot.update_buffers(sim_dt)
    if simulation_app.is_running():
        print('running')
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if simulation_app.is_running():
            print('running')
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            print('pause')
            sim.play()
            continue
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
        sim.step(render=not args_cli.headless)
        # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            # update marker positions
        sim.step()
        


if __name__ == "__main__":
    # Run empty stage
    main()
    # Close the simulator
    simulation_app.close()
