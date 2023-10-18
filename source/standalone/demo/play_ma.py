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
import scipy.ndimage as ndimage
from scipy import signal
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
from omni.isaac.core.prims import RigidPrim,GeometryPrim
from omni.isaac.orbit.utils.array import convert_to_torch

from pxr import Gf, UsdGeom

from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.cloner import GridCloner
# from omni.isaac.sensor import Camera
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
from omni.isaac.core.objects import DynamicCuboid
from PIL import Image
import matplotlib.pyplot as plt
import omni.isaac.core.utils.semantics as semantics_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
import asyncio
from omni.isaac.orbit.objects.rigid import RigidObject, RigidObjectCfg
from omni.isaac.orbit.utils.math import convert_quat, quat_mul, random_yaw_orientation, sample_cylinder
import scipy.spatial.transform as tf
"""
Main
"""
def load_ycb():
    ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }
    ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle']
    for key, usd_path in ycb_usd_paths.items():
        translation = torch.rand(3).tolist()
        translation = [translation[0],0.6*translation-0.3,0.01]
        prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation)
    # Setup rigid object
    cfg = RigidObjectCfg()
    # -- usd path
    cfg.meta_info.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
    # -- rotate the object to align with the ground plane
    cfg.init_state.rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90, 90, 0), degrees=True).as_quat(), to="wxyz")

    # Create rigid object handler
    rigid_object = RigidObject(cfg)

    # Spawn rigid object
    # note: Spawning object like this will apply rigid object properties and physics material configurations.
    rigid_object.spawn("/World/Objects/crackerBox2")

def point_cloud_process(pcd):

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    # o3d.visualization.draw_geometries([inlier_cloud])
    print(plane_model)
    return plane_model

def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    # sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01)
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch",device='cuda:0')
    # Set main camera
    set_camera_view([0, 2, 3.], [0.0, 0.0, 0])
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
    # prim_utils.create_prim("/World/Table", usd_path=table_path,position=(0,0,-0.25),scale=(1,0.6,0.5))
    Table = DynamicCuboid(prim_path="/World/Table",position=(0,0,-0.25),scale=(1,0.6,0.5))
    
    Table.set_mass(10000000) 
    sideTable = DynamicCuboid(prim_path="/World/sideTable",position=(0.35,-0.9,-0.3),scale=(0.4,0.4,0.4))
    sideTable.set_mass(10)
    #################### robot base
    prim_utils.create_prim("/World/Robotbase", usd_path=table_path,position=(0,-0.45,-0.2),scale=(0.3,0.26,0.4))
    #################### ycb path
    ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        "mug":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/025_mug.usd",
        "largeMarker":f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/040_large_marker.usd",
    }
    ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker']
    
    
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
    # image_pixel = [512,256]
    # cam = rep.create.camera(position=(0,0,5), look_at=(0,0,0))
    # rp = rep.create.render_product(cam, image_pixel)
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
    hand_camera.spawn("/World/Robot/panda_hand/hand_camera", translation=(0.1, 0.0, 0.0),orientation=(0,0,1,0))
    # Spawn camera
    camera.spawn("/World/CameraSensor")
    sim.reset()
    
    # rigid_object.initialize("/World/Objects/.*")
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
    # rgb_image = rep.AnnotatorRegistry.get_annotator("rgb")
    # depth_image = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    # camera_params = rep.AnnotatorRegistry.get_annotator("CameraParams")
    # point_cloud_get = rep.AnnotatorRegistry.get_annotator("pointcloud",init_params={"includeUnlabelled": True}).attach(rp)
    # rcp = camera_params.attach(rp)
    # rgb_image.attach(rp)
    # depth_image.attach(rp)

    for _ in range(100):
        sim.render()
    ##################################################################### load ycb
    for key, usd_path in ycb_usd_paths.items():
        
        translation = torch.rand(3).tolist()
        translation = [translation[0]*0.8-0.4,0.45*translation[1]-0.225,0.06]
        # translation = [translation[0]*0.8-0.4,0.8*translation[1]+0.5,0.1]
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,translation[0]*90), degrees=True).as_quat(), to="wxyz")
        if key in ["mug","tomatoSoupCan"]:
           rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,0,translation[0]*90), degrees=True).as_quat(), to="wxyz")
        prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
        GeometryPrim(f"/World/Objects/{key}",collision=True)
        RigidPrim(f"/World/Objects/{key}",mass=0.5)
    ##################################################################### 
    
    # rep.orchestrator.step()
    # rgb = rgb_image.get_data()
    # # rep.orchestrator.step()
    # depth = depth_image.get_data()
    # print(depth)
    # # rep.orchestrator.step()
    # point_cloud = point_cloud_get.get_data()
    # rcp_data = rcp.get_data()
    # print(point_cloud)
    
    ##################################################################### 
    print("[INFO]: Setup complete...")
    
    # points = []
    # points_rgb = []
    # points.append(point_cloud["data"])
    # points_rgb.append(point_cloud["info"]["pointRgb"].reshape(-1, 4)[:, :3])
    # pc_data = np.concatenate(points)
    # pc_rgb = np.concatenate(points_rgb)
    # # print(pc_rgb)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_data)
    # # pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    # o3d.visualization.draw_geometries([pcd])
    # # print(np.asarray(pcd.colors))
    # ply_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
    # os.makedirs(ply_out_dir, exist_ok=True)
    # o3d.io.write_point_cloud(os.path.join(ply_out_dir, "pointcloud.ply"), pcd)
    # # print(rgb.shape)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # point_cloud_process(pcd)
    # img = Image.fromarray(rgb[:,:,:3])
    # # ax.scatter(pc_data[:,0], pc_data[:,1], pc_data[:,2])
    # # plt.show()
    # plt.imshow(img)
    # plt.show()
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
        # print('robot action')
        # print(robot_actions)
        robot.apply_action(robot_actions)
        # perform step
        
        ##################################################################### 
        print("[INFO]: Setup complete...")
        
        
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            count +=1
            # update marker positions
        if count >=50:
            count = 0
            camera.update(dt=0.0)
            hand_camera.update(dt=0.0)
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
            select_m = np.dot(pointcloud_w,plane_model) - float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m >=-0.02)).reshape(-1).astype(int)
            inliers = pointcloud_w[index_inliers]
            select_m = np.dot(inliers,plane_model) - float(plane_model_ori[3])
            index_inliers = np.argwhere((select_m <=0.3)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            index_inliers = np.argwhere((inliers[:,1]>=-0.3)).reshape(-1).astype(int)
            inliers = inliers[index_inliers]
            # print(camera.data.output["distance_to_image_plane"].shape)
            # print(pointcloud_w.shape)
            select_m = np.dot(inliers,plane_model) - float(plane_model_ori[3])
            index_objects = np.argwhere((select_m>=0.01)).reshape(-1).astype(int)
            objects_point = inliers[index_objects].copy()
            objects_pcd = o3d.geometry.PointCloud()
            objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
            # o3d.visualization.draw_geometries([objects_pcd])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inliers)
            # o3d.visualization.draw_geometries([pcd])
            pts_tab = np.array(pcd.points)
            Nx,Ny = 100,60
            x = np.linspace(np.min(pts_tab[:,0]), np.max(pts_tab[:,0]), Nx)
            y = np.linspace(np.min(pts_tab[:,1]), np.max(pts_tab[:,1]), Ny)
            xv, yv = np.meshgrid(x, y)
            pts = np.array(objects_pcd.points)
            u = (pts[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
            v = (pts[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
            u = 99*u
            v = 59*v
            occupancy = np.zeros( (Ny,Nx) )
            u = u.astype(int)
            v = v.astype(int)
            u_ind = np.where(u<100)
            u = u[u_ind]
            v = v[u_ind]
            v_ind = np.where(v<60)
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
            # np.save("occu",occupancy)
            # structure = np.ones((3,3))
            # occupancy = ndimage.binary_dilation(occupancy,structure=structure)
            # grad = boundary_detection(occupancy)
            bound_detect(occupancy)
            rgb=camera.data.output["rgb"]
            rgb = convert_to_torch(rgb, device=sim.device, dtype=torch.float32)
            rgb = rgb[:, :, :3].cpu().data.numpy()
            # plt.imshow(grad)
            # plt.show()
            img = Image.fromarray((rgb).astype(np.uint8))
            hand_rgb=hand_camera.data.output["rgb"]
            hand_rgb = convert_to_torch(hand_rgb, device=sim.device, dtype=torch.float32)
            hand_rgb = hand_rgb[:, :, :3].cpu().data.numpy()
            place_new_object(occupancy,ycb_name,ycb_usd_paths,num_new)
            num_new +=1
            hand_img = Image.fromarray((hand_rgb).astype(np.uint8))
            for _ in range(25):
                sim.render()
                sim.step()
                sim.play()
                camera.update(dt=0.0)
                hand_camera.update(dt=0.0)
                
            plt.imshow(img)
            plt.show()
            # plt.imshow(hand_img)
            # plt.show()
            if num_new>=1:
                aabb_points = get_new_obj_pcd(hand_camera,(40,40),hand_plane_model)
            # bbox = rep_annotator.get_data()
            # print(bbox)
        sim.step()
def get_new_obj_pcd(camera,size,hand_plane_model):
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
    # select_m = np.dot(pointcloud_w,plane_model) - float(plane_model_ori[3])
    # index_inliers = np.argwhere((select_m >=-0.1)).reshape(-1).astype(int)
    # inliers = pointcloud_w[index_inliers]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(outlier_cloud)
    o3d.visualization.draw_geometries([inlier_cloud])
    o3d.visualization.draw_geometries([outlier_cloud])
    # select_m = np.dot(inliers,plane_model) - float(plane_model_ori[3])
    # index_inliers = np.argwhere((select_m <=0.3)).reshape(-1).astype(int)
    # inliers = inliers[index_inliers]
    # index_inliers = np.argwhere((inliers[:,1]>=-0.3)).reshape(-1).astype(int)
    # inliers = inliers[index_inliers]
    # # print(camera.data.output["distance_to_image_plane"].shape)
    # # print(pointcloud_w.shape)
    select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
    index_objects = np.argwhere((select_m>=0.01)).reshape(-1).astype(int)
    objects_point = pointcloud_w[index_objects].copy()
    # objects_pcd = outlier_cloud.select_by_index(inliers,invert = True)
    objects_pcd = o3d.geometry.PointCloud()
    objects_pcd.points = o3d.utility.Vector3dVector(objects_point)
    o3d.visualization.draw_geometries([objects_pcd])
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
    return aabb_points
def place_new_object(occu,ycb_list,ycb_path,num_new):
    random_i = random.randint(0,len(ycb_list)-1)
    translation = [0.35,-0.9,0.05]
    # translation = [translation[0]*0.8-0.4,0.8*translation[1]+0.5,0.1]
    key = ycb_list[random_i]
    # key = 'largeMarker'
    usd_path = ycb_path[key]
    if key in ["mug","tomatoSoupCan"]:
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,0,0), degrees=True).as_quat(), to="wxyz")
    else:  
        rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,0), degrees=True).as_quat(), to="wxyz")
    prim_utils.create_prim(f"/World/newObjects/{key+str(num_new)}", usd_path=usd_path, translation=translation,orientation=rot)
    GeometryPrim(f"/World/newObjects/{key+str(num_new)}",collision=True)
    RigidPrim(f"/World/newObjects/{key+str(num_new)}",mass=0.5)
    # prim_utils.delete_prim(f"/World/newObjects/{key+str(num_new)}")
    
def bound_detect(occu):
    occu_p = occu.copy()/np.max(occu)
    occu_p = np.array(occu*255,dtype=np.uint8)
    data = im.fromarray(occu_p) 
    data.save('occu.png') 
    img = cv2.imread('occu.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:",len(contours))
    shape_dict = dict()
    i = 0
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
    plt.imshow(occu)
    plt.show()
def boundary_detection(occu):
    kernal = np.ones((3,3))
    kernal = kernal *float(1./9.)
    grad = signal.convolve2d(occu,kernal,mode='same')
    index_bound = np.argwhere(grad>0)
    # print(index_bound)
    index_bound = index_bound.reshape((-1,2))
    # print(index_bound)
    grad_bound = grad[index_bound[:,0],index_bound[:,1]]
    # print(grad_bound)
    index2 = np.argwhere(grad_bound<0.4).reshape((-1)).astype(int)
    # print(index2)
    index_bound = index_bound[index2]
    grad_temp = np.zeros(occu.shape)
    # print('conv shape')
    # print(grad_temp.shape)
    # print(grad.shape)
    grad_temp[index_bound[:,0],index_bound[:,1]] = 1
    # print(index_bound)
    # plt.imshow(grad_temp)
    # plt.show()
    # grad[index_bound[:,0],index_bound[:,1]] = 2
    return grad_temp
def find_edges(points=np.array):
    points_tmp = points.copy()
    for i in range(len(points)):
        [point_x,point_y] = points_tmp[i]

if __name__ == "__main__":
    # Run empty stage
    main()
    # Close the simulator
    simulation_app.close()
