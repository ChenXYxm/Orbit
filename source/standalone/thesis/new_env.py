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


#################
from place_new_obj import place_new_obj_fun
"""
Main
"""


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

def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch",device='cuda:0')
    # Set main camera
    set_camera_view([0, 2, 3.], [0.0, 0.0, 0])
    # Enable GPU pipeline and flatcache
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
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
    Table = FixedCuboid(prim_path="/World/Table",position=(0,0,-0.25),scale=(1,0.6,0.5))
    # Table.set_mass(10000000) 
    sideTable = FixedCuboid(prim_path="/World/sideTable",position=(0.35,-0.9,-0.3),scale=(0.4,0.4,0.4))
    # sideTable.set_mass(10)
    #################### robot base
    prim_utils.create_prim("/World/Robotbase", usd_path=table_path,position=(0,-0.45,-0.2),scale=(0.3,0.26,0.4))
    #################### ycb path
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
        # "pitcherBase":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
        "bowl":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd",
        "largeClamp":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/051_large_clamp.usd",
        "scissors":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/037_scissors.usd",
    }
    ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
                'banana','bowl','largeClamp','scissors']
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
    sim.reset()
    ##########################################
    ##########################################
    for _ in range(14):
        sim.render()
    hand_camera.initialize()
    camera.initialize()
    robot.initialize()
    ik_controller.initialize()
    # Reset states
    robot.reset_buffers()
    ik_controller.reset_idx()
    position = [0, 0, 2.]
    orientation = [0, 0, -1, 0]
    camera.set_world_pose_ros(position, orientation)
    hand_camera.set_world_pose_ros([0.35,-0.9,0.8], orientation)
    Table.initialize()
    sideTable.initialize()
    sideTable.set_collision_enabled(True)
    Table.set_collision_enabled(True)

    for _ in range(10):
        sim.render()
    ##################################################################### load ycb
    
    obj_dict = dict()
    for _ in range(1):
        randi = np.random.randint(0,len(ycb_name))
        angle = np.random.randint(0,180)
        # angle = 0
        key_ori = ycb_name[randi]
        # key_ori = "mug"
        usd_path = ycb_usd_paths[key_ori]
        if key_ori not in obj_dict:
            obj_dict[key_ori] = 1
        else:
            obj_dict[key_ori] +=1
        key = key_ori+str(obj_dict[key_ori])
        translation = torch.rand(3).tolist()
        translation = [-translation[0]*0.3+0.2,-0.45*translation[1]-0.3,-0.2]
        # translation = [0,0,0.2]
        print(translation,angle,key_ori)
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
        if key_ori in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
            rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,angle,0), degrees=True).as_quat(), to="wxyz")
        prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
        GeometryPrim(f"/World/Objects/{key}",collision=True)
        RigidPrim(f"/World/Objects/{key}",mass=0.3)
        for _ in range(30):
            sim.step()
    num_obj = np.random.randint(0,5)
    if num_obj >=1:
        for _ in range(num_obj):
            randi = np.random.randint(0,len(ycb_name))
            angle = np.random.randint(0,180)
            # angle = 0
            key_ori = ycb_name[randi]
            # key_ori = "mug"
            usd_path = ycb_usd_paths[key_ori]
            if key_ori not in obj_dict:
                obj_dict[key_ori] = 1
            else:
                obj_dict[key_ori] +=1
            key = key_ori+str(obj_dict[key_ori])
            translation = torch.rand(3).tolist()
            translation = [translation[0]*0.8-0.4,0.45*translation[1]-0.225,0.1]
            # translation = [0,0,0.2]
            print(translation,angle,key_ori)
            rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
            if key_ori in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,angle,0), degrees=True).as_quat(), to="wxyz")
            prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
            GeometryPrim(f"/World/Objects/{key}",collision=True)
            RigidPrim(f"/World/Objects/{key}",mass=0.3)
            for _ in range(50):
                sim.step()
    ##################################################################### 
    print("[INFO]: Setup complete...")
    
    camera.update(dt=0.0)
    pcd = get_pcd(camera)
    # o3d.visualization.draw_geometries([pcd])
    plane_model = point_cloud_process(pcd)
    # Create buffers to store actions
    ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)
    if simulation_app.is_running():
        print('running')
    # Set end effector goals
    # Define goals for the arm
    ee_goals = [0.2, -0.95, 1.0, 0.0, 0, 0.0, 0]
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
    count = 0
    plane_model_ori = plane_model
    plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]])
    # rep_camera = rep.create.render_product("/World/CameraSensor",resolution=(640,480))
    # rep_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_loose")
    # rep_annotator.attach([rep_camera])
    num_new = 0
    hand_plane_model = None
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
        ########################################## ik control
        # # perform step
        # # set the controller commands
        # ik_controller.set_command(ik_commands)
        # # compute the joint commands
        # robot_actions[:, : robot.arm_num_dof] = ik_controller.compute(
        #     robot.data.ee_state_w[:, 0:3],
        #     robot.data.ee_state_w[:, 3:7],
        #     robot.data.ee_jacobian,
        #     robot.data.arm_dof_pos,
        # )
        # # in some cases the zero action correspond to offset in actuators
        # # so we need to subtract these over here so that they can be added later on
        # arm_command_offset = robot.data.actuator_pos_offset[:, : robot.arm_num_dof]
        # # offset actuator command with position offsets
        # # note: valid only when doing position control of the robot
        # robot_actions[:, : robot.arm_num_dof] -= arm_command_offset
        # # apply actions
        # # print('robot action')
        # # print(robot_actions)
        # robot.apply_action(robot_actions)
        # # perform step
        
        ##################################################################### 
        print("[INFO]: Setup complete...")
        
        
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            count +=1
            # update marker positions
        if count >=10:
            count = 0
            camera.update(dt=0.0)
            hand_camera.update(dt=0.0)
            pcd = get_pcd(camera)
            pointcloud_w = np.array(pcd.points)
            select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m >=-0.01)).reshape(-1).astype(int)
            inliers = pointcloud_w[index_inliers]
            select_m = np.dot(inliers,plane_model) - float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m <=0.3)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,1]>=-0.3)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            # print(camera.data.output["distance_to_image_plane"].shape)
            # print(pointcloud_w.shape)
            select_m = np.dot(inliers,plane_model) - float(plane_model_ori[3])
            index_objects = np.argwhere((select_m>=0.005)).reshape(-1).astype(int)
            objects_point = inliers[index_objects].copy()
            objects_pcd = o3d.geometry.PointCloud()
            objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
            # o3d.visualization.draw_geometries([objects_pcd])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inliers)
            # o3d.visualization.draw_geometries([pcd])
            pts_tab = np.array(pcd.points)
            Nx,Ny = 200,120
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
            # plt.imshow(occupancy)
            # plt.show()
            #
            # bound_detect(occupancy)
            # rgb=camera.data.output["rgb"]
            # rgb = convert_to_torch(rgb, device=sim.device, dtype=torch.float32)
            # rgb = rgb[:, :, :3].cpu().data.numpy()
            # plt.imshow(grad)
            # plt.show()
            # img = Image.fromarray((rgb).astype(np.uint8))
            # hand_rgb=hand_camera.data.output["rgb"]
            # hand_rgb = convert_to_torch(hand_rgb, device=sim.device, dtype=torch.float32)
            # hand_rgb = hand_rgb[:, :, :3].cpu().data.numpy()
            
            obj_dict, new_obj,obj_type,new_obj_path = place_new_object(occupancy,ycb_name,ycb_usd_paths,num_new,obj_dict)
            num_new +=1
            for _ in range(50):
                sim.step()
            # hand_img = Image.fromarray((hand_rgb).astype(np.uint8))   
            # plt.imshow(img)
            # plt.show()
            # plt.imshow(hand_img)
            # plt.show()
            # if num_new>=1:
                # aabb_points = get_new_obj_pcd(hand_camera,(40,40),hand_plane_model)
            aabb_points,_,vertices_new_obj = get_new_obj_info(hand_camera,(80,80),hand_plane_model)
            print(occupancy.shape)
            flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(occupancy,vertices_new_obj)
            if flag_found:
                # for i in range(len(new_poly_vetices)):
                #     occu_tmp[int(new_poly_vetices[i][1]),int(new_poly_vetices[i][0])] = 3
                # plt.imshow(occu_tmp)
                # plt.show()
                prim_utils.delete_prim(new_obj_path)
                if obj_type in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,np.rad2deg(new_obj_pos[2]),0), degrees=True).as_quat(), to="wxyz")
                else:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                # new_obj.set_default_state(position=[(50-new_obj_pos[1])*0.01,(new_obj_pos[0]-30)*0.01,0.2],orientation=rot)
                
                # new_obj.set_world_pose(position=[(50-new_obj_pos[1])*0.01,(new_obj_pos[0]-30)*0.01,0.2],orientation=rot)
                # new_obj.initialize()
                print(new_obj_pos)
                translation = [(Nx/2-new_obj_pos[1])*1./Nx,(new_obj_pos[0]-Ny/2)*1./Nx,0.1]
                print(translation)
                usd_path = ycb_usd_paths[obj_type]
                prim_utils.create_prim(new_obj_path, usd_path=usd_path, position=translation,orientation=rot)
                new_obj = GeometryPrim(new_obj_path,collision=True)
                RigidPrim(new_obj_path,mass=0.3)
                for _ in range(50):
                    sim.step
            # bbox = rep_annotator.get_data()
            # print(bbox)
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
    # o3d.visualization.draw_geometries([pcd])
    return pcd
def get_new_obj_info(camera,size,hand_plane_model):
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
    # plt.imshow(occupancy)
    # plt.show()
    vertices_new_obj = get_new_obj_contour_bbox(occupancy)
    return aabb_points,occupancy, vertices_new_obj
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
        print(vertices_new_obj)
        vertices_new_obj = vertices_new_obj - np.array([Nx/2,Ny/2])
        print(vertices_new_obj)
        plt.imshow(mask_tmp)
        plt.show()
        
        l = []
        for i in range(2):
            l.append(np.linalg.norm(vertices_new_obj[i]-vertices_new_obj[i+1]))
        print(l)
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
    print(np.array(aabb.get_box_points()))
    aabb_points = np.array(aabb.get_box_points()).reshape((-1,3))
    aabb.color = (1, 0, 0)
    o3d.visualization.draw_geometries([objects_pcd, aabb])
    return aabb_points
def place_new_object(occu,ycb_list,ycb_path,num_new,obj_dict):
    randi = np.random.randint(0,len(ycb_list))
    # angle = np.random.randint(0,180)
    angle = 0
    key_ori = ycb_list[randi]
    usd_path = ycb_path[key_ori]
    if key_ori not in obj_dict:
        obj_dict[key_ori] = 1
    else:
        obj_dict[key_ori] +=1
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
    main()
    # Close the simulator
    simulation_app.close()
