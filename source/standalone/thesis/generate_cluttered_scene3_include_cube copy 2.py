# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
''' different items'''
"""This script demonstrates how to create a simple stage in Isaac Sim with lights and a ground plane."""

"""Launch Isaac Sim Simulator first."""

import pickle
import argparse
import os
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
from PIL import Image as im 
# import os
import torch
import cv2
import numpy as np
import open3d as o3d
import random
from scipy import signal
from omni.isaac.core.prims import RigidPrim,GeometryPrim
from omni.isaac.orbit.utils.array import convert_to_torch
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
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.orbit.objects import RigidObjectCfg,RigidObject
from PIL import Image
import matplotlib.pyplot as plt
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
import asyncio
from omni.isaac.orbit.objects.rigid import RigidObject, RigidObjectCfg
from omni.isaac.orbit.utils.math import convert_quat
import scipy.spatial.transform as tf
from shapely import Polygon, STRtree, area, contains
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

import carb
#################
from source.standalone.thesis.place_new_obj_Feb16 import place_new_obj_fun
"""
Main
"""
class sim_cfg:
    sim: SimCfg = SimCfg(
            dt=0.01,
            substeps=4,
            physx=PhysxCfg(
                # num_position_iterations=8,
                gpu_max_rigid_contact_count=1024**2, #1024**2*2,
                gpu_max_rigid_patch_count=160*2048*10, #160*2048*10, #160*2048*10,
                gpu_found_lost_pairs_capacity = 1024 * 1024 * 2 * 2,#1024 * 1024 * 2 * 1, #1024 * 1024 * 2 * 8,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 32 * 1,#1024 * 1024 * 32 * 1, #1024 * 1024 * 32,
                gpu_total_aggregate_pairs_capacity=1024 * 1024 * 2 *1, #1024 * 1024 * 2 *1, #1024 * 1024 * 2 * 8
                friction_correlation_distance=0.0025,
                friction_offset_threshold=0.04,
                bounce_threshold_velocity=0.5,
                gpu_max_num_partitions=8,
                
                
            ),
            
        )

def point_cloud_process(pcd):

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    print(plane_model)
    return plane_model
def _configure_simulation_flags( sim_params: dict = None):
        """Configure simulation flags and extensions at load and run time."""
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings_iface.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx if requested
        # note: helpful when creating contact reporting over limited number of objects in the scene
        if sim_params["disable_contact_processing"]:
            carb_settings_iface.set_bool("/physics/disableContactProcessing", True)

    
def main():
    """Spawns lights in the stage and sets the camera view."""
    sim_params = sim_cfg.sim.to_dict()
    if sim_params is not None:
        if "physx" in sim_params:
            physx_params = sim_params.pop("physx")
            sim_params.update(physx_params)
    # set flags for simulator
    _configure_simulation_flags(sim_params)
    # create a simulation context to control the simulator
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.1,
        backend="torch",
        sim_params=sim_params,
        # physics_prim_path="/physicsScene",
        device='cuda:0',
    )
    # Load kit helper
    # sim = SimulationContext(physics_dt=0.01, rendering_dt=1, backend="torch",device='cuda:0',)
    # Set main camera
    set_camera_view([0, 2, 3.], [0.0, 0.0, 0])
    # Enable GPU pipeline and flatcache
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # PhysicsContext.set_gpu_total_aggregate_pairs_capacity(4000)
    # Enable hydra scene-graph instancing
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    ####################################################
    ####################################################
    ################# SCENE ############################
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
    Table = FixedCuboid(prim_path="/World/Table",position=(0,0,-0.25),scale=(0.5,0.5,0.5))
    # Table.set_mass(10000000) 
    sideTable = FixedCuboid(prim_path="/World/sideTable",position=(0.35,-0.9,-0.3),scale=(0.4,0.4,0.4))
    # sideTable.set_mass(10)
    #################### robot base
    prim_utils.create_prim("/World/Robotbase", usd_path=table_path,position=(0,-0.45,-0.2),scale=(0.3,0.26,0.4))
    #################### ycb path
    ######################################### load ycb objects
    # ycb_usd_paths = {
    #     "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    #     "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    #     "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    #     "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    #     "mug":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/025_mug.usd",
    #     "largeMarker":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/040_large_marker.usd",
    #     "tunaFishCan":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    #     "banana":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd",
    #     # "pitcherBase":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
    #     "bowl":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd",
    #     "largeClamp":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/051_large_clamp.usd",
    #     "scissors":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/037_scissors.usd",
    # }
    # ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
    #             'banana','bowl','largeClamp','scissors']
    # ycb_usd_paths = {
    #     "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    #     "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    #     # "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    #     "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    # }
    # ycb_name = ['crackerBox','sugarBox','mustardBottle']
    ycb_usd_paths = {
        # "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "Cube": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        
    }
    ycb_name = ['sugarBox','mustardBottle','tomatoSoupCan','Cube']
    obj1 = []
    obj_name_list = []
    for i in range(8):
        obj_cfg1 = RigidObjectCfg()
        obj_cfg1.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[0]],scale=(1.0, 1.0, 1.0),)
        obj_cfg1.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(2-0.25*i, 0.8, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        )
        obj_cfg1.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=0.5,
            max_linear_velocity=0.5,
            max_depenetration_velocity=0.5,
            disable_gravity=False,
        )
        obj_cfg1.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
        )
        obj1.append(RigidObject(obj_cfg1))
        obj_name_list.append(ycb_name[0]+str(i))
    obj2 = []
    for i in range(8):
        obj_cfg2 = RigidObjectCfg()
        obj_cfg2.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[1]],scale=(1.0, 1.0, 1.0),)
        obj_cfg2.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(2-0.25*i, 1.0, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        )
        obj_cfg2.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=0.5,
            max_linear_velocity=0.5,
            max_depenetration_velocity=0.5,
            disable_gravity=False,
        )
        obj_cfg2.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
        )
        obj2.append(RigidObject(obj_cfg2))
        obj_name_list.append(ycb_name[1]+str(i))
    obj3 = []
    for i in range(8):
        obj_cfg3 = RigidObjectCfg()
        obj_cfg3.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[2]],scale=(1.35,0.33,1.35),)
        obj_cfg3.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(2-0.25*i, 1.2, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        )
        obj_cfg3.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=0.5,
            max_linear_velocity=0.5,
            max_depenetration_velocity=0.5,
            disable_gravity=False,
        )
        obj_cfg3.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
        )
        obj3.append(RigidObject(obj_cfg3))
        obj_name_list.append(ycb_name[2]+str(i))
    obj4 = []
    for i in range(16):
        obj_cfg4 = RigidObjectCfg()
        obj_cfg4.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[3]],scale=(0.42,0.28,1.35),)
        obj_cfg4.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(2-0.25*i, 1.3, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        )
        obj_cfg4.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=0.5,
            max_linear_velocity=0.5,
            max_depenetration_velocity=0.5,
            disable_gravity=False,
        )
        obj_cfg4.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
        )
        obj4.append(RigidObject(obj_cfg4))
        obj_name_list.append(ycb_name[3]+str(i))
    # obj3 = []
    # for i in range(9):
    #     obj_cfg3 = RigidObjectCfg()
    #     obj_cfg3.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[2]],scale=(1.0, 1.0, 1.0),)
    #     obj_cfg3.init_state = RigidObjectCfg.InitialStateCfg(
    #     pos=(2-0.25*i, 1.2, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    #     )
    #     obj_cfg3.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
    #         solver_position_iteration_count=16,
    #         solver_velocity_iteration_count=1,
    #         max_angular_velocity=0.5,
    #         max_linear_velocity=0.5,
    #         max_depenetration_velocity=0.5,
    #         disable_gravity=False,
    #     )
    #     obj_cfg3.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
    #         static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    #     )
    #     obj3.append(RigidObject(obj_cfg3))
    #     obj_name_list.append(ycb_name[2]+str(i))
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
    ###################################### spawn items
    for i,obj_t in enumerate(obj1):
        obj_t.spawn(f"/World/Objs/obj1/obj_{i}")
    for i,obj_t in enumerate(obj2):
        obj_t.spawn(f"/World/Objs/obj2/obj_{i}")
    for i,obj_t in enumerate(obj3):
        obj_t.spawn(f"/World/Objs/obj3/obj_{i}")
    for i,obj_t in enumerate(obj4):
        obj_t.spawn(f"/World/Objs/obj4/obj_{i}")
    ###################################### sensor extension camera
    
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=240,
        width=240,
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device="cuda")
    # hand_camera = Camera(cfg=camera_cfg,device='cuda')
    # hand_camera.spawn("/World/Robot/panda_hand/hand_camera", translation=(0.1, 0.0, 0.0),orientation=(0,0,1,0))
    # hand_camera.spawn("/World/hand_camera")
    # Spawn camera
    camera.spawn("/World/CameraSensor")
    sim.reset()
    ##########################################
    ##########################################
    for _ in range(14):
        sim.render()
    # hand_camera.initialize()
    camera.initialize()
    robot.initialize()
    ik_controller.initialize()
    for i,obj_t in enumerate(obj1):
            obj_t.initialize(f"/World/Objs/obj1/obj_{i}")
    for i,obj_t in enumerate(obj2):
            obj_t.initialize(f"/World/Objs/obj2/obj_{i}")
    for i,obj_t in enumerate(obj3):
            obj_t.initialize(f"/World/Objs/obj3/obj_{i}")
    for i,obj_t in enumerate(obj4):
            obj_t.initialize(f"/World/Objs/obj4/obj_{i}")
    # Reset states
    robot.reset_buffers()
    ik_controller.reset_idx()
    position = [0, 0, 1.2]
    orientation = [0, 0, -1, 0]
    camera.set_world_pose_ros(position, orientation)
    # hand_camera.set_world_pose_ros([0.35,-0.9,0.55], orientation)
    Table.initialize()
    sideTable.initialize()
    sideTable.set_collision_enabled(True)
    Table.set_collision_enabled(True)
    for i,obj_t in enumerate(obj1):
            obj_t.update_buffers(0.01)
    for i,obj_t in enumerate(obj2):
            obj_t.update_buffers(0.01)
    for i,obj_t in enumerate(obj3):
            obj_t.update_buffers(0.01)
    for i,obj_t in enumerate(obj4):
            obj_t.update_buffers(0.01)
    for _ in range(10):
        sim.render()
    ##################################################################### get plane model of the table
    camera.update(dt=0.0)
    pcd = get_pcd(camera)
    plane_model = point_cloud_process(pcd)
    plane_model_ori = plane_model
    plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]])
    ##################################################################### load ycb
    obj_dict = dict()
    table_obj_pos_rot = dict()
    for i in range(1000):
        print('i')
        print(i)
        ## initialize parameters
        for _ in ycb_name:
            obj_dict[_] = 0
        table_obj_pos_rot = dict()
        ################## random drop one item
        randi = np.random.randint(0,len(ycb_name))
        obj_name_i = ycb_name[randi]
        angle = np.random.randint(0,180)
        translation = torch.rand(2).tolist()
        translation = [0.33*translation[0]-0.165,0.33*translation[1]-0.165,0.07]
        if obj_name_i in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
            rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,0,0), degrees=True).as_quat(), to="wxyz")
        else:
            rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
                
        # rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
        if randi == 0: 
            root_state = obj1[0].get_default_root_state()
            root_state[:,:3] = torch.tensor(translation).cuda()
            root_state[:,3:7] = torch.tensor(rot).cuda()
            obj1[0].set_root_state(root_state)
            obj1[0].update_buffers(0.01)
        elif randi == 1:
            root_state = obj2[0].get_default_root_state()
            root_state[:,:3] = torch.tensor(translation).cuda()
            root_state[:,3:7] = torch.tensor(rot).cuda()
            obj2[0].set_root_state(root_state)
            obj2[0].update_buffers(0.01)
        elif randi == 2:
            root_state = obj3[0].get_default_root_state()
            root_state[:,:3] = torch.tensor(translation).cuda()
            root_state[:,3:7] = torch.tensor(rot).cuda()
            obj3[0].set_root_state(root_state)
            obj3[0].update_buffers(0.01)
        elif randi == 3:
            root_state = obj4[0].get_default_root_state()
            root_state[:,:3] = torch.tensor(translation).cuda()
            root_state[:,3:7] = torch.tensor(rot).cuda()
            obj4[0].set_root_state(root_state)
            obj4[0].update_buffers(0.01)
        table_obj_pos_rot[obj_name_i] = [(translation,rot)]
        
        print('new obj to be placed, obj_id,current_num')
        print(obj_name_i,randi,obj_dict[obj_name_i])
        for j in range(10):
            sim.step()
        obj_dict[obj_name_i] += 1 
        #################### start the loop to drop item iteratively
        while obj_dict[ycb_name[0]] < 8 or obj_dict[ycb_name[1]] < 8 or obj_dict[ycb_name[2]] < 8 or obj_dict[ycb_name[3]]<16 :
            # print(j)
            if simulation_app.is_running():
                print('running')
            if sim.step(render=not args_cli.headless):
                print('pause')
                sim.play()
                continue
            camera.update(dt=0.01)
            for _ in range(10):
                sim.render()
            camera.update(dt=0.01)
            pcd = get_pcd(camera)
            pointcloud_w = np.array(pcd.points)
            select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m >=-0.01)).reshape(-1).astype(int)
            inliers = pointcloud_w[index_inliers]
            select_m = np.dot(inliers,plane_model) + float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m <=0.3)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,1]>=-0.25)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,1]<=0.25)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,0]>=-0.25)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,0]<=0.25)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            select_m = np.dot(inliers,plane_model) + float(plane_model_ori[3])
            index_objects = np.argwhere((select_m>=0.001)).reshape(-1).astype(int)
            objects_point = inliers[index_objects].copy()
            objects_pcd = o3d.geometry.PointCloud()
            objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
            # o3d.visualization.draw_geometries([objects_pcd])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inliers)
            # o3d.visualization.draw_geometries([pcd])
            pts_tab = np.array(pcd.points)
            Nx,Ny =50,50
            x = np.linspace(np.min(pts_tab[:,0]), np.max(pts_tab[:,0]), Nx)
            y = np.linspace(np.min(pts_tab[:,1]), np.max(pts_tab[:,1]), Ny)
            xv, yv = np.meshgrid(x, y)
            pts = np.array(objects_pcd.points)
            u = (pts[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
            v = (pts[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
            u = (Nx-1)*u
            v = (Ny-1)*v
            occupancy = np.zeros( (Ny,Nx) )
            u = u.astype(int)
            v = v.astype(int)
            u_ind = np.where(u<Nx)
            u = u[u_ind]
            v = v[u_ind]
            v_ind = np.where(v<Ny)
            u = u[v_ind]
            v = v[v_ind]
            u_ind = np.where(u>=0)
            u = u[u_ind]
            v = v[u_ind]
            v_ind = np.where(v>=0)
            u = u[v_ind]
            v = v[v_ind]
            occupancy[v,u] = 1
            occupancy = np.fliplr(occupancy)

            randi = np.random.randint(0,len(ycb_name))
            obj_name_i = ycb_name[randi]
            while obj_dict[obj_name_i] >=8:
                randi = np.random.randint(0,len(ycb_name))
                obj_name_i = ycb_name[randi]
            vertices_new_obj = get_new_obj_info(obj_name_i)
            print('new obj to be placed, obj_id,current_num')
            print(obj_name_i,randi,obj_dict[obj_name_i])

            flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(occupancy,vertices_new_obj)
            if flag_found:
                obj_dict[obj_name_i] += 1 
                if obj_name_i in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,np.rad2deg(new_obj_pos[2]),0), degrees=True).as_quat(), to="wxyz")
                else:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                # print(new_obj_pos)
                translation = [(Nx/2-new_obj_pos[1])*1./100.,(new_obj_pos[0]-Ny/2)*1./100.,0.05]
                # print(translation)
                if randi == 0:
                    j = int(obj_dict[obj_name_i]-1)
                    root_state = obj1[j].get_default_root_state()
                    root_state[:,:3] = torch.tensor(translation).cuda()
                    root_state[:,3:7] = torch.tensor(rot).cuda()
                    obj1[j].set_root_state(root_state)
                    for k in range(30):
                        sim.step()
                    obj1[j].update_buffers(0.01)
                    
                    print('current state,obj_num')
                    print(obj1[j].data.root_pos_w,j)
                elif randi == 1:
                    j = int(obj_dict[obj_name_i]-1)
                    root_state = obj2[j].get_default_root_state()
                    root_state[:,:3] = torch.tensor(translation).cuda()
                    root_state[:,3:7] = torch.tensor(rot).cuda()
                    obj2[j].set_root_state(root_state)
                    for k in range(30):
                        sim.step()
                    obj2[j].update_buffers(0.01)
                    print('current state,obj_num')
                    print(obj2[j].data.root_pos_w,j)
                elif randi == 2:
                    j = int(obj_dict[obj_name_i]-1)
                    root_state = obj3[j].get_default_root_state()
                    root_state[:,:3] = torch.tensor(translation).cuda()
                    root_state[:,3:7] = torch.tensor(rot).cuda()
                    obj3[j].set_root_state(root_state)
                    for k in range(30):
                        sim.step()
                    obj3[j].update_buffers(0.01)
                    print('current state,obj_num')
                    print(obj3[j].data.root_pos_w,j)
                elif randi == 3:
                    j = int(obj_dict[obj_name_i]-1)
                    root_state = obj4[j].get_default_root_state()
                    root_state[:,:3] = torch.tensor(translation).cuda()
                    root_state[:,3:7] = torch.tensor(rot).cuda()
                    obj4[j].set_root_state(root_state)
                    for k in range(30):
                        sim.step()
                    obj4[j].update_buffers(0.01)
                    print('current state,obj_num')
                    print(obj4[j].data.root_pos_w,j)
                if obj_name_i not in table_obj_pos_rot:
                    table_obj_pos_rot[obj_name_i] = [(translation,rot)]
                else:
                    table_obj_pos_rot[obj_name_i].append((translation,rot))
                for k in range(20):
                    sim.step()
                for k in range(20):
                    sim.render()
                # print('current state')
                # print(obj1[j].data.root_pos_w)
            else:
                flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(occupancy,vertices_new_obj)
                if flag_found:
                    obj_dict[obj_name_i] += 1 
                    if obj_name_i in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                        rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,np.rad2deg(new_obj_pos[2]),0), degrees=True).as_quat(), to="wxyz")
                    else:
                        rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                    # print(new_obj_pos)
                    translation = [(Nx/2-new_obj_pos[1])*1./100.,(new_obj_pos[0]-Ny/2)*1./100.,0.05]
                    # print(translation)
                    if randi == 0:
                        j = int(obj_dict[obj_name_i]-1)
                        root_state = obj1[j].get_default_root_state()
                        root_state[:,:3] = torch.tensor(translation).cuda()
                        root_state[:,3:7] = torch.tensor(rot).cuda()
                        obj1[j].set_root_state(root_state)
                        for k in range(30):
                            sim.step()
                        obj1[j].update_buffers(0.01)
                        
                        print('current state,obj_num')
                        print(obj1[j].data.root_pos_w,j)
                    elif randi == 1:
                        j = int(obj_dict[obj_name_i]-1)
                        root_state = obj2[j].get_default_root_state()
                        root_state[:,:3] = torch.tensor(translation).cuda()
                        root_state[:,3:7] = torch.tensor(rot).cuda()
                        obj2[j].set_root_state(root_state)
                        for k in range(30):
                            sim.step()
                        obj2[j].update_buffers(0.01)
                        print('current state,obj_num')
                        print(obj2[j].data.root_pos_w,j)
                    elif randi == 2:
                        j = int(obj_dict[obj_name_i]-1)
                        root_state = obj3[j].get_default_root_state()
                        root_state[:,:3] = torch.tensor(translation).cuda()
                        root_state[:,3:7] = torch.tensor(rot).cuda()
                        obj3[j].set_root_state(root_state)
                        for k in range(30):
                            sim.step()
                        obj3[j].update_buffers(0.01)
                        print('current state,obj_num')
                        print(obj3[j].data.root_pos_w,j)
                    elif randi == 3:
                        j = int(obj_dict[obj_name_i]-1)
                        root_state = obj4[j].get_default_root_state()
                        root_state[:,:3] = torch.tensor(translation).cuda()
                        root_state[:,3:7] = torch.tensor(rot).cuda()
                        obj4[j].set_root_state(root_state)
                        for k in range(30):
                            sim.step()
                        obj4[j].update_buffers(0.01)
                        print('current state,obj_num')
                        print(obj4[j].data.root_pos_w,j)
                    if obj_name_i not in table_obj_pos_rot:
                        table_obj_pos_rot[obj_name_i] = [(translation,rot)]
                    else:
                        table_obj_pos_rot[obj_name_i].append((translation,rot))
                    for k in range(20):
                        sim.step()
                    for k in range(20):
                        sim.render()
                else:
                    file_name_ori = "dict_"
                    file_list = os.listdir("train_table4/")
                    print('done')
                    print(table_obj_pos_rot)
                    for k in table_obj_pos_rot:
                        print(len(table_obj_pos_rot[k]))
                    num_file = 1
                    while True:
                        file_name = file_name_ori+str(num_file)+".pkl"
                        if file_name in file_list:
                            num_file +=1
                        else:
                            file_path = "train_table4/"+file_name
                            f_save = open(file_path,'wb')
                            table_obj_pos_rot = [table_obj_pos_rot,obj_name_i]
                            pickle.dump(table_obj_pos_rot,f_save)
                            f_save.close()
                            break
                    # f_save = open('generated_table/dict_file.pkl','wb')
                    # pickle.dump(table_obj_pos_rot,f_save)
                    # f_save.close()
                    break
        #################### reset
        for i,obj_t in enumerate(obj1):
            root_state = obj_t.get_default_root_state()
            obj_t.set_root_state(root_state)
        for i,obj_t in enumerate(obj2):
            root_state = obj_t.get_default_root_state()
            obj_t.set_root_state(root_state)
        for i,obj_t in enumerate(obj3):
            root_state = obj_t.get_default_root_state()
            obj_t.set_root_state(root_state)
        for i,obj_t in enumerate(obj4):
            root_state = obj_t.get_default_root_state()
            obj_t.set_root_state(root_state)
        for j in range(30):
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
    # rgb=camera.data.output["rgb"]
    # rgb = convert_to_torch(rgb, device='cuda:0', dtype=torch.float32)
    # rgb = rgb[:, :, :3].cpu().data.numpy()
    
    # img = Image.fromarray((rgb).astype(np.uint8))
    # plt.imshow(img)
    # plt.show()
    # o3d.visualization.draw_geometries([pcd])
    return pcd
def get_new_obj_info(obj_type):
# def get_new_obj_info(camera,size,hand_plane_model,obj_type):
    '''
    pcd = get_pcd(camera)
    # o3d.visualization.draw_geometries([pcd])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.005,
                                         ransac_n=3,
                                         num_iterations=1000)
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    plane_model_ori = plane_model
    plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]])
    pointcloud_w = np.array(outlier_cloud.points)
    
    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])
    
    select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
    index_objects = np.argwhere((select_m>=0.01)).reshape(-1).astype(int)
    objects_point = pointcloud_w[index_objects].copy()
    # objects_pcd = outlier_cloud.select_by_index(inliers,invert = True)
    objects_pcd = o3d.geometry.PointCloud()
    objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
    # o3d.visualization.draw_geometries([objects_pcd])
    pcd = inlier_cloud
    # o3d.visualization.draw_geometries([pcd])
    pts_tab = np.array(pcd.points)
    # pts_tab[:,2] = 0
    # pcd.points = o3d.utility.Vector3dVector(pts_tab)
    aabb = objects_pcd.get_oriented_bounding_box()
    print(np.array(aabb.get_box_points()))
    aabb_points = np.array(aabb.get_box_points()).reshape((-1,3))
    aabb.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([objects_pcd, aabb])
    Nx = size[1]
    Ny = size[0]
    x = np.linspace(np.min(pts_tab[:,0]), np.max(pts_tab[:,0]), Nx)
    y = np.linspace(np.min(pts_tab[:,1]), np.max(pts_tab[:,1]), Ny)
    xv, yv = np.meshgrid(x, y)
    pts = np.array(objects_pcd.points)
    u = (pts[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
    v = (pts[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
    u = (size[0]-1)*u
    v = (size[1]-1)*v
    occupancy = np.zeros( (Ny,Nx) )
    u = u.astype(int)
    v = v.astype(int)
    u_ind = np.where(u<size[0])
    u = u[u_ind]
    v = v[u_ind]
    v_ind = np.where(v<size[1])
    u = u[v_ind]
    v = v[v_ind]
    u_ind = np.where(u>=0)
    u = u[u_ind]
    v = v[u_ind]
    v_ind = np.where(v>=0)
    u = u[v_ind]
    v = v[v_ind]
    occupancy[v,u] = 1
    occupancy = np.fliplr(occupancy)
    file_list = os.listdir("obj_mask/")
    file_name = obj_type +"_mask.pkl"
    if file_name not in file_list:
        file_path = "obj_mask/"+file_name
        f_save = open(file_path,'wb')
        pickle.dump(occupancy,f_save)
        f_save.close()
    '''
    occupancy = np.zeros((40,40))
    file_list = os.listdir("obj_mask/")
    for i in range(len(file_list)):
        if obj_type in file_list[i]:
            fileObject2 = open('obj_mask/'+file_list[i], 'rb')
            occupancy=  pickle.load(fileObject2)
            fileObject2.close()
    # plt.imshow(occupancy)
    # plt.show()
    vertices_new_obj = get_new_obj_contour_bbox(occupancy)
    return vertices_new_obj
def get_new_obj_contour_bbox(occu:np.array):
    mask = occu.copy()
    shape_occu = occu.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    mask = np.array((mask-np.min(mask))*255/(np.max(mask)-np.min(mask)),dtype=np.uint8)
    ret,mask = cv2.threshold(mask,50,255,0)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    cnt = []
    for i in contours:
        area_tmp = cv2.contourArea(i)
        if area_tmp>max_area:
            max_area = area_tmp
            cnt = i
    # approx = cv2.minAreaRect(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    # print(approx)
    # box = cv2.boxPoints(approx)
    # approx = np.int0(box)
    if x+w >=occu.shape[1]:
        w = occu.shape[1]-x-1
    if y+h >=occu.shape[0]:
        h = occu.shape[0]-1-y
    approx = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    vertices_new_obj = []
    mask_tmp = mask.copy()
    if len(approx) >=2:
        approx = approx.reshape((-1,2))
        for i in range(len(approx)):
            mask_tmp[approx[i][1],approx[i][0]] = 130
        vertices_new_obj = approx 
        # print(vertices_new_obj)
        vertices_new_obj = vertices_new_obj - np.array([Nx/2,Ny/2])
        # print(vertices_new_obj)
        # plt.imshow(mask_tmp)
        # plt.show()
        
        l = []
        for i in range(2):
            l.append(np.linalg.norm(vertices_new_obj[i]-vertices_new_obj[i+1]))
        # print(l)
        return vertices_new_obj
    else:
        return None
def get_new_obj_pcd(camera,size,hand_plane_model):
    pcd = get_pcd(camera)
    # o3d.visualization.draw_geometries([pcd])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.005,
                                         ransac_n=3,
                                         num_iterations=1000)
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    plane_model_ori = plane_model
    plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]])
    pointcloud_w = np.array(outlier_cloud.points)
    
    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])
    
    select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
    index_objects = np.argwhere((select_m>=0.01)).reshape(-1).astype(int)
    objects_point = pointcloud_w[index_objects].copy()
    # objects_pcd = outlier_cloud.select_by_index(inliers,invert = True)
    objects_pcd = o3d.geometry.PointCloud()
    objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
    # o3d.visualization.draw_geometries([objects_pcd])
    pcd = inlier_cloud
    o3d.visualization.draw_geometries([pcd])
    pts_tab = np.array(pcd.points)
    # pts_tab[:,2] = 0
    # pcd.points = o3d.utility.Vector3dVector(pts_tab)
    aabb = objects_pcd.get_oriented_bounding_box()
    # print(np.array(aabb.get_box_points()))
    aabb_points = np.array(aabb.get_box_points()).reshape((-1,3))
    aabb.color = (1, 0, 0)
    o3d.visualization.draw_geometries([objects_pcd, aabb])
    return aabb_points
def place_new_object(occu,ycb_list,ycb_path,num_new,obj_dict):
    angle = 0
    while True:
        randi = np.random.randint(0,len(ycb_list))
        # angle = np.random.randint(0,180)
        key_ori = ycb_list[randi]
        usd_path = ycb_path[key_ori]
        if key_ori not in obj_dict:
            obj_dict[key_ori] = 1
        else:
            if obj_dict[key_ori]<8:
                obj_dict[key_ori] +=1
                break
            
    print(obj_dict)
    key = key_ori+str(obj_dict[key_ori])
    translation = 0
    translation = [0.35,-0.9,0.1]
    rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
    if key_ori in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,angle,0), degrees=True).as_quat(), to="wxyz")
    prim_utils.create_prim(f"/World/newObjects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
    new_obj = GeometryPrim(f"/World/newObjects/{key}",collision=True)
    RigidPrim(f"/World/newObjects/{key}",mass=0.3)
    new_obj_path = f"/World/newObjects/{key}"
    return obj_dict,new_obj,key_ori,new_obj_path
    # prim_utils.delete_prim(f"/World/newObjects/{key+str(num_new)}")
    
def bound_detect(occu):
    occu_p = occu.copy()/np.max(occu)
    occu_p = np.array(occu*255,dtype=np.uint8)
    # data = im.fromarray(occu_p) 
    # data.save('pic/occu.png') 
    # img = cv2.imread('pic/occu.png')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(occu_p,50,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:",len(contours))
    shape_dict = dict()
    i = 0
    polygons = []
    for cnt in contours:
        i += 1
        # approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        # approx = cv2.convexHull(cnt)
        approx = cv2.minAreaRect(cnt)
        (x,y) = cnt[0,0]
        # print(approx)
        box = cv2.boxPoints(approx)
        approx = np.int0(box)
        # print(box)
        if len(approx) >=2:
            # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)
            # print(approx)
            approx = np.array(approx).reshape((-1,2))
            shape_dict[i] = approx
            polygons.append(Polygon(approx))
    print(polygons)
    tree = STRtree(polygons)
    print("remove containing boxes")
    del_ind = []
    for i,poly in enumerate(polygons):
        indice = tree.query(poly, predicate="contains").tolist()
        print(i)
        print(tree.geometries.take(indice))
        print(tree.geometries.take(i))
        if len(indice) >0:
            for j in indice:
                if contains(poly,tree.geometries.take(j)) and area(poly)>area(tree.geometries.take(j)):
                    if j+1 in shape_dict:
                        del(shape_dict[j+1])
                        del_ind.append(int(j))
    print(del_ind)
    for i in shape_dict:
        points_tmp = shape_dict[i].copy()
        points_tmp[:,0] = points_tmp[:,1]
        points_tmp[:,1] = shape_dict[i][:,0].copy()
        for j in range(len(points_tmp)):
            p_s = points_tmp[j]
            if j < len(points_tmp)-1:
                p_e = points_tmp[j+1]
            else:
                p_e = points_tmp[0]
            line = p_e - p_s
            length = np.linalg.norm(line)
            # print(p_s,p_e,length)
            for k in range(int(np.ceil(length))):
                if np.ceil(p_s[0]+k*line[0]/length) < occu.shape[0] and np.ceil(p_s[1]+k*line[1]/length)<occu.shape[1] and np.ceil(p_s[1]+k*line[1]/length)>=0 and np.ceil(p_s[0]+k*line[0]/length)>=0:
                    occu[int(np.ceil(p_s[0]+k*line[0]/length)),int(np.ceil(p_s[1]+k*line[1]/length))] = 2
    polygons = []
    for i in shape_dict:
        polygons.append(shape_dict[i])
    print(polygons)
    plt.imshow(occu)
    plt.show()
    return shape_dict



if __name__ == "__main__":
    # Run empty stage
    # for i in range(3):
        # simulation_app = SimulationApp(config)
        main()
        # Close the simulator
        simulation_app.close()
