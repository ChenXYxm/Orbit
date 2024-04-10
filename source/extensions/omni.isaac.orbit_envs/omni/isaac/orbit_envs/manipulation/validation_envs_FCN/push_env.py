# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import gym.spaces
import math
import torch
import cv2
from typing import List
import numpy as np
import open3d as o3d
import warp as wp
import omni.isaac.core.utils.prims as prim_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
import pickle as pkl
from matplotlib import cm
from omni.isaac.core.materials import PhysicsMaterial
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, convert_quat,random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.core.objects import FixedCuboid
from .push_cfg import PushEnvCfg, RandomizationCfg, YCBobjectsCfg, CameraCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.core.prims import RigidPrim,GeometryPrim
# from omni.isaac.orbit.utils.math import 
from omni.isaac.orbit.utils.array import convert_to_torch
import scipy.spatial.transform as tf
from .place_new_obj import place_new_obj_fun,get_new_obj_contour_bbox,draw_bbox,placing_compare_fun_2
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.orbit.objects import RigidObjectCfg
import torchvision
class PushEnv(IsaacEnv):
    """Environment for lifting an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: PushEnvCfg = None, **kwargs):
        self.step_num = 0
        # copy configuration
        self.reset_f = False
        self.show_first_og = True
        self.cfg = cfg
        self.obj_name_list = []
        self.obj_handle_list = []
        ycb_usd_paths = self.cfg.YCBdata.ycb_usd_paths
        ycb_name = self.cfg.YCBdata.ycb_name
        self.place_success_all = 0.0
        self.reaching_all = 0.0
        self.step_all = 0.0
        self.fallen_all = 0.0
        self.un_satisfied_scenes = []
        self.obj_list = []
        
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        self.obj_handle_list = [f'handle_{i}' for i in range(int(1*len(ycb_name)))]
        # print(self.obj_handle_list)
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.obj1 = []
        
        self.pushing_policy_result = dict()
        for i in range(9):
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
            self.obj1.append(RigidObject(obj_cfg1))
            self.obj_name_list.append(ycb_name[0]+str(i))
        self.obj2 = []
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
            self.obj2.append(RigidObject(obj_cfg2))
            self.obj_name_list.append(ycb_name[1]+str(i))
        self.obj3 = []
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
            self.obj3.append(RigidObject(obj_cfg3))
            self.obj_name_list.append(ycb_name[2]+str(i))
        self.obj4 = []
        for i in range(16):
            obj_cfg4 = RigidObjectCfg()
            obj_cfg4.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[3]],scale=(0.5,1,1.2),)
            obj_cfg4.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(2-0.1*i, 1.3, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
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
            self.obj4.append(RigidObject(obj_cfg4))
            self.obj_name_list.append(ycb_name[3]+str(i))
        self.obj5 = []
        for i in range(16):
            obj_cfg5 = RigidObjectCfg()
            obj_cfg5.meta_info = RigidObjectCfg.MetaInfoCfg(usd_path=ycb_usd_paths[ycb_name[4]],scale=(0.5,0.6,1.2),)
            obj_cfg5.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(2-0.25*i, 1.6, -0.4), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
            )
            obj_cfg5.rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=0.5,
                max_linear_velocity=0.5,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            )
            obj_cfg5.physics_material = RigidObjectCfg.PhysicsMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
            )
            self.obj5.append(RigidObject(obj_cfg5))
            self.obj_name_list.append(ycb_name[4]+str(i))
        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()
        self.table_scenes = [0 for i in range(self.num_envs)]
        self.push_out = [0 for i in range(self.num_envs)]
        # prepare the observation manager
        self._observation_manager = PushObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = PushRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        self.new_obj_mask = np.zeros((self.num_envs,self.cfg.og_resolution.new_obj[1],self.cfg.og_resolution.new_obj[0]))
        
        # print information about MDP
        # print('this is toy v12')
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"]
        print("num_obs")
        print(num_obs)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=num_obs,dtype=np.uint8)
        # compute the action space
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        # self.action_space = gym.spaces.Box(low=np.array([-0.5,-0.3,0]), high=np.array([0.5,0.3,1]), dtype=np.float64)
        
        self.action_space = gym.spaces.MultiDiscrete([self.cfg.og_resolution.tabletop[0],
                                                      self.cfg.og_resolution.tabletop[1]])
        
        # self.action_space = gym.spaces.MultiDiscrete([3,3])
        # self.action_space = gym.spaces.MultiDiscrete(low=0.0,high = 1.0,shape=(self.cfg.og_resolution.tabletop[0],
        #                                               self.cfg.og_resolution.tabletop[1],8))
        print("[INFO]: Completed setting up the environment...")
        
        # Take an initial step to initialize the scene.
        # This is required to compute quantities like Jacobians used in step()
        self.new_obj_vertices = []
        self.sim.step()
        # -- fill up buffers
        
        self.robot.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj1):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj2):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj3):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj4):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj5):
            obj_t.update_buffers(self.dt)
        self.obj_on_table = dict()
        self.obj_on_table_name = dict()
    """
    Implementation specifics.
    """
    
    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-.5)
        # table
        table_size = (float(self.cfg.og_resolution.tabletop[0])/100.0,float(self.cfg.og_resolution.tabletop[1])/100.0,0.5)
        self.Table = FixedCuboid(self.template_env_ns + "/Table",position=(0,0,-0.25),scale=table_size)
        self.Table.set_collision_enabled(True)
        self.Table.set_collision_approximation("convexDecomposition")
        prim_path = self.template_env_ns + "/Table"
        if self.cfg.object.physics_material is not None:
            # -- resolve material path
            material_path = self.cfg.object.physics_material.prim_path
            if not material_path.startswith("/"):
                material_path = prim_path + "/" + prim_path
            # -- create physics material
            material = PhysicsMaterial(
                prim_path=material_path,
                static_friction=self.cfg.object.physics_material.static_friction,
                dynamic_friction=self.cfg.object.physics_material.dynamic_friction,
                restitution=self.cfg.object.physics_material.restitution,
            )
            # -- apply physics material
            kit_utils.apply_nested_physics_material(prim_path, material.prim_path)
        
        prim_utils.create_prim(self.template_env_ns + "/Robotbase", usd_path=self.cfg.table.table_path,position=(0,-0.45,-0.2),scale=(0.3,0.26,0.4))
        
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot",translation=(0.0, -.52, -0.1))
        # object
        # self.object.spawn(self.template_env_ns + "/Object")
        for i,obj_t in enumerate(self.obj1):
            obj_t.spawn(self.template_env_ns + f"/Objs/obj1/obj_{i}")
        for i,obj_t in enumerate(self.obj2):
            obj_t.spawn(self.template_env_ns + f"/Objs/obj2/obj_{i}")
        for i,obj_t in enumerate(self.obj3):
            obj_t.spawn(self.template_env_ns + f"/Objs/obj3/obj_{i}")
        for i,obj_t in enumerate(self.obj4):
            obj_t.spawn(self.template_env_ns + f"/Objs/obj4/obj_{i}")
        for i,obj_t in enumerate(self.obj5):
            obj_t.spawn(self.template_env_ns + f"/Objs/obj5/obj_{i}")
        # camera
        position_camera = [0, 0, 1.2]
        orientation = [1, 0, 0, 0]
        
        self.camera = Camera(cfg=self.cfg.camera.camera_cfg, device='cuda')
        # Spawn camera
        self.camera.spawn(self.template_env_ns + "/CameraSensor",translation=position_camera,orientation=orientation)
        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            # self._goal_markers = StaticMarker(
            #     "/Visuals/object_goal",
            #     self.num_envs,
            #     usd_path=self.cfg.goal_marker.usd_path,
            #     scale=self.cfg.goal_marker.scale,
            # )
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current",
                self.num_envs,
                usd_path=self.cfg.frame_marker.usd_path,
                scale=self.cfg.frame_marker.scale,
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command",
                    self.num_envs,
                    usd_path=self.cfg.frame_marker.usd_path,
                    scale=self.cfg.frame_marker.scale,
                )
          
        # return list of global prims
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        # self.step_num = 0
        if not self.reset_f:
            self.new_obj_type = [i for i in range(self.num_envs)]
            self.env_i_tmp = 47
        self.reset_f = True
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)

        ''' modified for toy example
        # self.reset_objs(env_ids=env_ids)
        # self._randomize_table_scene(env_ids=env_ids)
        '''
        '''modified for toy example v2'''
        self.reset_objs(env_ids=env_ids)
        # print()
        # self.new_obj_mask = np.zeros((self.num_envs,self.cfg.og_resolution.new_obj[1],self.cfg.og_resolution.new_obj[0]))
        # for i in range(100):
        #     self.sim.step(render=self.enable_render)
        for i in env_ids.tolist():
            env_ids_tmp = torch.from_numpy(np.array([i])).to(self.device)
            self._randomize_table_scene(env_ids=env_ids_tmp)
            
        # print(self.table_scenes)
        '''modified for toy example v2'''
        
        for _ in range(30):
            self.sim.step()
        self._update_table_og()
        self.table_og_pre[env_ids] = self.table_og[env_ids].clone()
        self.table_expand_og_pre[env_ids] = self.table_expand_og[env_ids].clone()
        self.table_tsdf_pre[env_ids] = self.table_tsdf[env_ids].clone()

        # self.falling_obj[env_ids] = 0
        self.falling_obj_all[env_ids] = 0
        
        self._check_fallen_objs(env_ids)
         
        self.show_first_og = True
       
        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.actions_origin[env_ids] = 0
        self.actions_ori[env_ids] = 0
        self.actions[env_ids] = 0
        # self.falling_obj[env_ids] = 0
        # self.stop_pushing[env_ids] = 0
        # self.falling_obj_all[env_ids] = 0
        self.place_success[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.step_count[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)
        ############## get the info of new obj
        self._get_obj_mask(env_ids=env_ids)
        self._get_obj_info(env_ids=env_ids)
        self.generate_patch_img(env_ids=env_ids)
        self.pushing_policy_result[self.env_i_tmp][5] = self.table_expand_og[0].clone().detach().cpu().numpy()

    def _get_obj_mask(self,env_ids: VecEnvIndices):
        if np.sum(self.new_obj_mask) ==0:
            # print('no mask')
            self.new_obj_mask[env_ids.tolist()] = self.cfg.obj_mask.mask['sugarBox'] 
        # print(self.new_obj_mask.shape)
        mask = np.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],self.cfg.og_resolution.tabletop[0]))
        s_x_ind = int(self.cfg.og_resolution.tabletop[1]/2-self.new_obj_mask.shape[2]/2)
        e_x_ind = int(self.cfg.og_resolution.tabletop[1]/2+self.new_obj_mask.shape[2]/2)
        s_y_ind = int(self.cfg.og_resolution.tabletop[0]/2-self.new_obj_mask.shape[1]/2)
        e_y_ind = int(self.cfg.og_resolution.tabletop[0]/2+self.new_obj_mask.shape[1]/2)
        mask[:,s_x_ind:e_x_ind,s_y_ind:e_y_ind] = self.new_obj_mask
        for j in env_ids.tolist():
            # print("mask j")
            # print(env_ids[j])
            self.obj_masks[j] = torch.from_numpy(mask[j]).to(self.device)
            # plt.imshow(self.new_obj_mask[j])
            # plt.draw()
            # plt.imshow(mask[j])
            # plt.show() 
    ''' function only for toy example'''
    def generate_patch_img(self,env_ids):
        # self.random_patches = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
        #                              self.cfg.og_resolution.tabletop[0]),device=self.device)
        rand_i = np.random.randint(0,high=30,size=(self.num_envs,2))
        # rand_i = rand_i.reshape(-1)
        # print(env_ids)
        for i in env_ids.tolist():
            # print(i)
            self.random_patches[i]  = 0
            start_i_x = int((rand_i[i,0])+8)
            end_i_x = int((rand_i[i,0])+16)
            start_i_y = int((rand_i[i,1])+8)
            end_i_y = int((rand_i[i,1])+16)
            self.random_patches[i][start_i_x:end_i_x,start_i_y:end_i_y] = 1
            # plt.imshow(self.random_patches[i].cpu().numpy())
            # plt.show()
    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.push_out = [0 for i in range(self.num_envs)]
        self.stop_pushing[:] = 0
        self.step_all += 1.0
        self.previous_actions = self.actions_origin.clone()
        self.table_og_pre = self.table_og.clone()
        self.table_tsdf_pre = self.table_tsdf.clone()
        self.table_expand_og_pre = self.table_expand_og.clone()
        
        self.step_num +=1
        self.step_count[:] +=1
        
        # print(self.previous_actions)
        # print(self.actions_origin)
        #self.actions = actions.clone().to(device=self.device)
        ''' modified for toy example
        self.actions_origin = actions.clone()
        '''
        ''' modified for toy example'''
        # print('actions')
        # print(actions)
        self.actions_origin = actions.detach().clone().long()
        # print('actions')
        # print(self.actions_origin)
        self.action_ori = actions.detach().clone()
        ########### test
        # x_tmp = np.random.randint(0,50)
        # y_tmp = np.random.randint(0,50)
        # self.actions_origin[:,0] = x_tmp
        # self.action_ori[:,0] = x_tmp
        # self.actions_origin[:,1] = y_tmp
        # self.action_ori[:,1] = y_tmp
        ########### test
        '''toy version 1
        # self.actions_origin += 4*(actions.clone()-torch.ones((self.num_envs,self.num_actions)).to(self.device))
        # for i in range(self.num_envs):
        #     self.actions_origin[i,0] = max(0,int(self.actions_origin[i,0]))
        #     self.actions_origin[i,0] = min(self.cfg.og_resolution.tabletop[0]-1,int(self.actions_origin[i,0]))
        #     self.actions_origin[i,1] = max(0,int(self.actions_origin[i,1]))
        #     self.actions_origin[i,1] = min(self.cfg.og_resolution.tabletop[1]-1,int(self.actions_origin[i,1]))
        '''
        ###################### transform discrete actions into start positions and pushing directions
        # print(self.new_obj_type)
        self._check_reaching_toy_v2()
        flag_placed = self._check_placing()
        self._update_table_og()
        # print(flag_placed,self.new_obj_type,self.check_reaching)
        # print('action')
        # print(actions)
        # print(self.check_reaching,flag_placed)
        # print(self.actions_origin)
        
        ''' modified for toy example'''
        
        # self.actions = actions.detach().clone()
        # self.actions = self.actions.type(torch.float16)
        # actions_tmp = self.actions.clone()
        # self.actions[:,2] = self.actions[:,2]/8.0
        # action_range = (float(self.cfg.og_resolution.tabletop[0])/200.0,float(self.cfg.og_resolution.tabletop[1])/200.0)
        # self.actions[:,1] = action_range[1]*(actions_tmp[:,0].clone()-float(self.cfg.og_resolution.tabletop[1]/2))/float(self.cfg.og_resolution.tabletop[1]/2)
        # self.actions[:,0] = action_range[0]*(-actions_tmp[:,1].clone()+float(self.cfg.og_resolution.tabletop[0]/2))/float(self.cfg.og_resolution.tabletop[0]/2)
        start_t = int((self.cfg.og_resolution.expand_obs[1]-self.cfg.og_resolution.tabletop[1])/2)
        self.actions = actions.detach().clone()
        self.actions = self.actions.type(torch.float16)
        self.actions[:,:2] = self.actions[:,:2]-start_t
        actions_tmp = self.actions.clone()
        self.actions[:,2] = self.actions[:,2]/8.0
        
        action_range = (float(self.cfg.og_resolution.tabletop[0])/200.0,float(self.cfg.og_resolution.tabletop[1])/200.0)
        self.actions[:,1] = action_range[1]*(actions_tmp[:,0].clone()-float(self.cfg.og_resolution.tabletop[1]/2))/float(self.cfg.og_resolution.tabletop[0]/2)
        self.actions[:,0] = action_range[0]*(-actions_tmp[:,1].clone()+float(self.cfg.og_resolution.tabletop[0]/2))/float(self.cfg.og_resolution.tabletop[0]/2)
        # print(self.actions)
        # print(self.actions)
        ################# stop pushing
        # for i in range(self.num_envs):
        #     if self.check_reaching[i] == 0:
        #                 # print("stop actions")
        #                 # print(self.actions)
        #                 self.stop_pushing[i] = 1  
        #                 self.actions[i,1] = -0.5
        #                 self.actions[i,0] = 0.5
        #                 self.actions[i,2] = 0
        for i in range(self.num_envs):
            if self.cfg.flag_compare:
                self.check_reaching[i] = 1
            if self.actions[i,2]>1.0 and not flag_placed:
                # print('stop action')
                # print(self.actions[i],i,actions[i])
                # print(actions[i])
                self.actions[i,1] = -0.5
                self.actions[i,0] = 0.5
                self.actions[i,2] = 0
                self.stop_pushing[i] = 1
            if self.check_reaching[i] == 0:
                        
                self.actions[i,1] = -0.5
                self.actions[i,0] = 0.5
                self.actions[i,2] = 0
            if self.check_reaching[i] == 0 and not flag_placed:
                self.stop_pushing[i] = 1
            if self.stop_pushing[i] == 1 or flag_placed:
                self.check_reaching[i] = 0

        actions_tmp = torch.zeros((self.num_envs,self._ik_controller.num_actions),device=self.device)
        actions_tmp[:,:2] = self.actions[:,:2].clone()
        actions_tmp[:,:3] = self.actions[:,:3].clone()
        ##################### make the palm of the manipulator pependicular to the pushing direction
        for i in range(self.num_envs):
            if actions_tmp[i,2]<0.5:
                rot = convert_quat(tf.Rotation.from_euler("XYZ", (180,0,-360*float(actions_tmp[i,2].cpu())), degrees=True).as_quat(), to="wxyz")
            else:
                rot = convert_quat(tf.Rotation.from_euler("XYZ", (180,0,-360*float(actions_tmp[i,2].cpu()-0.5)), degrees=True).as_quat(), to="wxyz")
            
            actions_tmp[i,3:7] = torch.from_numpy(rot).to(self.device)
        # actions_tmp[:,:3] = self.actions.clone()
        # actions_tmp[:,1] +=0.1
        ''' modified for toy example'''
        if not self.cfg.pre_train:
            ########### lift the gripper above the start position
            if torch.sum(self.check_reaching)>0:
                self.pushing_policy_result[self.env_i_tmp][2] += 1
                for i in range(25):
                    self.robot.update_buffers(self.dt)
                    
                    if self.cfg.control.control_type == "inverse_kinematics":
                        
                        actions_tmp[:,2] = 0.3
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # close the gripper
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(27):
                    self.robot.update_buffers(self.dt)
                    
                    if self.cfg.control.control_type == "inverse_kinematics":
                        
                        actions_tmp[:,2] = 0.12
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # close the gripper
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(8):
                    self.robot.update_buffers(self.dt)
                    
                    if self.cfg.control.control_type == "inverse_kinematics":
                        
                        actions_tmp[:,2] = 0.06
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # close the gripper
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                # print(self.robot.data.ee_state_w[:, 0:3])
                # if self.cfg.viewer.debug_vis and self.enable_render:
                #     self._debug_vis()
                ############ let the gripper go down to the start position
                
                for i in range(20):
                    self.robot.update_buffers(self.dt)
                    
                    if self.cfg.control.control_type == "inverse_kinematics":
                        
                        actions_tmp[:,2] = 0.02
                        
                        self._ik_controller.set_command(actions_tmp[:, :])
                        
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                ################# perform pushing
                for i in range(self.num_envs):
                    vec_tmp = np.zeros(2)
                    if not self.cfg.flag_compare:
                        vec_tmp[0] = 0.035*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                        vec_tmp[1] = 0.035*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    else:
                        vec_tmp[0] = 0.1*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                        vec_tmp[1] = 0.1*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    actions_tmp[i,:2] = actions_tmp[i,:2] + torch.from_numpy(vec_tmp).to(self.device)
                for i in range(6):
                    self.robot.update_buffers(self.dt)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        self._ik_controller.set_command(actions_tmp[:, :])
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(self.num_envs):
                    vec_tmp = np.zeros(2)
                    if not self.cfg.flag_compare:
                        vec_tmp[0] = 0.035*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                        vec_tmp[1] = 0.035*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    else:
                        vec_tmp[0] = 0.1*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                        vec_tmp[1] = 0.1*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[0] = 0.035*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[1] = 0.035*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[0] = 0.1*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[1] = 0.1*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    actions_tmp[i,:2] = actions_tmp[i,:2] + torch.from_numpy(vec_tmp).to(self.device)
                for i in range(6):
                    self.robot.update_buffers(self.dt)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                    
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()   
                ################ lift the gripper
                for i in range(self.num_envs):
                    vec_tmp = np.zeros(2)
                    vec_tmp[0] = 0.02*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                    vec_tmp[1] = 0.02*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[0] = 0.1*np.cos(2*np.pi*self.actions[i,2].cpu().numpy())
                    # vec_tmp[1] = 0.1*np.sin(2*np.pi*self.actions[i,2].cpu().numpy())
                    actions_tmp[i,:2] = actions_tmp[i,:2] - torch.from_numpy(vec_tmp).to(self.device)
                for i in range(2):
                    self.robot.update_buffers(self.dt)
                    # print("robot dof pos")
                    # print(self.robot.data.ee_state_w[:, 0:7])
                    # print(self.envs_positions)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.025
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(3):
                    self.robot.update_buffers(self.dt)
                    # print("robot dof pos")
                    # print(self.robot.data.ee_state_w[:, 0:7])
                    # print(self.envs_positions)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.045
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(3):
                    self.robot.update_buffers(self.dt)
                    # print("robot dof pos")
                    # print(self.robot.data.ee_state_w[:, 0:7])
                    # print(self.envs_positions)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.07
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(3):
                    self.robot.update_buffers(self.dt)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.12
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(3):
                    self.robot.update_buffers(self.dt)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.2
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                for i in range(3):
                    self.robot.update_buffers(self.dt)
                    if self.cfg.control.control_type == "inverse_kinematics":
                        # set the controller commands
                        actions_tmp[:,2] = 0.4
                        # self.robot.data.ee_state_w[:, 0:7]
                        self._ik_controller.set_command(actions_tmp[:, :])
                        # self.actions[:, -1] = -0.1
                        # use IK to convert to joint-space commands
                        self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                            self.robot.data.ee_state_w[:, 3:7],
                            self.robot.data.ee_jacobian,
                            self.robot.data.arm_dof_pos,
                        )
                        # offset actuator command with position offsets
                        dof_pos_offset = self.robot.data.actuator_pos_offset
                        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
                        # we assume last command is tool action so don't change that
                        self.robot_actions[:, -1] = -1 # self.actions[:, -1]
                    elif self.cfg.control.control_type == "default":
                        self.robot_actions[:] = actions_tmp
                    # perform physics stepping
                    for _ in range(self.cfg.control.decimation):
                        # set actions into buffers
                        self.robot.apply_action(self.robot_actions)
                        # simulate
                        self.sim.step(render=self.enable_render)
                        # check that simulation is playing
                        if self.sim.is_stopped():
                            return
                    if self.cfg.viewer.debug_vis and self.enable_render:
                        self._debug_vis()
                # post-step:
                # -- compute common buffers
                for _ in range(10):
                    self.sim.step()
                
                env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
                dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
                self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
                self.robot.update_buffers(self.dt)
                for _ in range(110):
                    self.sim.step()
                
                env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
                dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
                self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
                self.robot.update_buffers(self.dt)
                for _ in range(70):
                    self.sim.step()
        env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
        for i,obj_t in enumerate(self.obj1):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj2):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj3):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj4):
            obj_t.update_buffers(self.dt)
        for i,obj_t in enumerate(self.obj5):
            obj_t.update_buffers(self.dt)
        ''' modified for toy example
            self._check_fallen_objs(env_ids)
            # check_placing
            self._check_placing()
        '''
        ''' remove useless scene'''
        # self._check_placing()
        
        # if not self.cfg.pre_train:
        #     self._check_fallen_objs(env_ids)
        #     self._check_placing()
        # else:
        #     self._update_table_og()
        
        if not self.cfg.pre_train:
            self._check_fallen_objs(env_ids)
            ############## TODO: changed in Feb 1
            self._check_placing() #delete this condition in Feb 1
            #self._update_table_og() # add this condition at Feb 1
            self._update_table_og()
        else: 
            self._update_table_og()
        # self._check_fallen_objs(env_ids)
        # self._update_table_og()
        for i in range(self.num_envs):
            if self.check_reaching[i] ==0:
                self.place_success[i] = 0
        # reward
        self.reward_buf = self._reward_manager.compute()
        # print("reward")
        # # # print(self._reward_manager.compute())
        # print(self.reward_buf)
        ##################### terminations #######################
        ''' only for toy example'''
        # self._check_reaching_toy_v2()
        ''' only for toy example'''
        self._check_termination()
        self.delta_same_action = torch.where(torch.sum(torch.abs(self.previous_actions[:,:3]-self.actions_origin),dim=1)<=0.1,1,0)

        # -- store history
        ''' modified for toy example
        self.previous_actions = self.actions.clone()
        '''
        
        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- add information to extra if task completed
        # object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
        '''modified for toy example
        self.extras["is_success"] = torch.where(torch.from_numpy(np.ones(self.num_envs)).to(self.device) < 0.02, 1, self.reset_buf)
        '''
        
        # print(self.extras["time_outs"])

        ''' modified for toy example'''
        # self.extras["is_success"] = torch.where(self.place_success>=0.5, 1, self.reset_buf) #changed in 08 Dec
        if not self.cfg.pre_train:
            tmp = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.extras["is_success"] = torch.where(self.place_success>=0.5, 1, tmp)
            # print(self.extras["time_outs"])
            for i,value_timeout in enumerate(self.extras['time_outs']):
                if value_timeout:
                    if self.place_success[i]>=0.5:
                        self.extras["time_outs"][i] = False
        else:
            tmp = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.extras["is_success"] = torch.where(self.check_reaching>=0.5, 1,tmp)
            # print(self.extras["time_outs"])
            for i,value_timeout in enumerate(self.extras['time_outs']):
                if value_timeout:
                    if self.check_reaching[i]>=0.5:
                        self.extras["time_outs"][i] = False
        ''' modified for toy example'''
        # -- update USD visualization
        # self._update_table_og()
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        # print("obser")
        # obser_tmp = self._observation_manager.compute()['policy']
        # for i in range(self.num_envs):
        #     plt.imshow(obser_tmp[i,:,:,8].cpu().numpy())
        #     plt.show()
        #     plt.imshow(obser_tmp[i,:,:,2].cpu().numpy())
        #     plt.show()
        # print(self._observation_manager.compute()['policy'])
        
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre-processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torchee
        # randomization
        # -- initial pose
        config = self.cfg.randomization.object_initial_pose
        for attr in ["position_uniform_min", "position_uniform_max"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))
        # -- desired pose
        config = self.cfg.randomization.object_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.cams = [self.camera] + [Camera(cfg=self.cfg.camera.camera_cfg,device='cuda') for _ in range(self.num_envs - 1)]
        
        self.sim.reset()
        
        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        for i,obj_t in enumerate(self.obj1):
            obj_t.initialize(self.env_ns + "/.*"+ f"/Objs/obj1/obj_{i}")
        for i,obj_t in enumerate(self.obj2):
            obj_t.initialize(self.env_ns + "/.*"+ f"/Objs/obj2/obj_{i}")
        for i,obj_t in enumerate(self.obj3):
            obj_t.initialize(self.env_ns + "/.*"+ f"/Objs/obj3/obj_{i}")
        for i,obj_t in enumerate(self.obj4):
            obj_t.initialize(self.env_ns + "/.*"+ f"/Objs/obj4/obj_{i}")
        for i,obj_t in enumerate(self.obj5):
            obj_t.initialize(self.env_ns + "/.*"+ f"/Objs/obj5/obj_{i}")
        self.new_obj_vertices = [i for i in range(self.num_envs)]
        
        for _ in range(15):
            self.sim.render()
        for i in range(self.num_envs):
           self.cams[i].initialize(self.env_ns + f"/env_{i}/CameraSensor/Camera")
           env_pos = np.array(self.envs_positions[i].cpu().numpy())
           self.cams[i].set_world_pose_from_view(eye=np.array([0, 0, 1.7]) + env_pos, target=np.array([0, 0, 0]) + env_pos)
           self.cams[i].update(self.dt)
        
        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            # self.num_actions = self._ik_controller.num_actions
            # self.num_actions = 3
            self.num_actions = 3
        elif self.cfg.control.control_type == "default":
            # self.num_actions = self.robot.num_actions
            self.num_actions = 3

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.actions_origin = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.actions_ori = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.object_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        # buffers
        self.object_root_pose_ee = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.table_og = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.table_og_pre = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.table_tsdf = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.table_tsdf_pre = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.table_expand_og = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1]+12,
                                     self.cfg.og_resolution.tabletop[0]+12),device=self.device)
        self.table_expand_og_pre = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1]+12,
                                     self.cfg.og_resolution.tabletop[0]+12),device=self.device)
        self.obj_masks = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.place_success = torch.zeros((self.num_envs,),device=self.device)
        self.falling_obj = torch.zeros((self.num_envs,),device=self.device)
        self.falling_obj_all = torch.zeros((self.num_envs,),device=self.device)
        self.step_count = torch.zeros((self.num_envs,),device=self.device)
        self.stop_pushing = torch.zeros((self.num_envs,),device=self.device)
        self.delta_same_action = torch.zeros((self.num_envs,),device=self.device)
        self.done_obs = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
        self.random_patches = torch.zeros((self.num_envs,self.cfg.og_resolution.tabletop[1],
                                     self.cfg.og_resolution.tabletop[0]),device=self.device)
    def get_pcd(self,camera):
        ############## get the pointcloud
        camera.update(dt=self.dt)
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
        # print('camera intrinsic',camera.data.intrinsic_matrix)
        # print('camera pos',camera.data.position,camera.data.orientation)
        # from scipy.spatial.transform import Rotation
        # r = Rotation.from_quat(camera.data.orientation)
        # extrinsic_matrix = np.eye(4)
        # extrinsic_matrix[:3, :3] = r.as_matrix()
        # extrinsic_matrix[:3, 3] = camera.data.position
        # print(extrinsic_matrix)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_w)
        # print(camera.data.output["distance_to_image_plane"].type())
        # _volume = o3d.pipelines.integration.UniformTSDFVolume(
        #     length=0.5,
        #     resolution=50,
        #     sdf_trunc=0.1,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        # )
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(
        #     width=250,
        #     height=250,
        #     intrinsic_matrix = camera.data.intrinsic_matrix
            
        # )
        # # print(camera.data.output["rgb"])
        # # print(np.array(camera.data.output["rgb"]))
        # rgb = camera.data.output["rgb"].numpy()
        # depth = camera.data.output["distance_to_image_plane"].numpy().reshape(250,250)
        # # print('depth shape',depth.shape)
        # # print(depth)
        # # if isinstance(rgb,np.ndarray):
        # #     print('the array is numpy')
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     o3d.geometry.Image(rgb),
        #     o3d.geometry.Image(depth),
        #     depth_scale=1.0,
        #     depth_trunc=2.0,
        #     convert_rgb_to_intensity=False,
        # )
        # _volume.integrate(rgbd, intrinsic, extrinsic_matrix)

        # cloud = _volume.extract_voxel_point_cloud()
        # points = np.asarray(cloud.points)
        # distances = np.asarray(cloud.colors)[:, [0]]
        # grid = np.zeros((1, 50, 50, 50), dtype=np.float32)
        # for idx, point in enumerate(points):
        #     i, j, k = np.floor(point / 0.01).astype(int)
        #     # print(i,j,k)
        #     # i += 25
        #     # j += 25
        #     grid[0, i, j, k] = distances[idx]
        # mesh = _volume.extract_triangle_mesh()
        # o3d.visualization.draw([mesh])
        # print(grid)

        # print('pointcloud 3d')
        # print(pointcloud_w)
        # o3d.visualization.draw_geometries([pcd])
        return pcd
    def point_cloud_process(self,pcd):
        ################ do plane detection
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
        return plane_model
    def add_salt_and_pepper_noise(self,image, salt_prob, pepper_prob):
        noisy_image = np.copy(image)

        
        num_salt = np.ceil(salt_prob * image.size)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 1

        
        num_pepper = np.ceil(pepper_prob * image.size)
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

    def get_og(self,camera):
        ############ get occupancy grid 
        camera.update(dt=0.0)
        pcd = self.get_pcd(camera)
        cam_pos = camera.data.position
        # print(cam_pos)
        # o3d.visualization.draw_geometries([pcd])
        plane_model = self.point_cloud_process(pcd)
        plane_model_ori = plane_model
        plane_model= np.array([plane_model[0],plane_model[1],plane_model[2]])
        pointcloud_w = np.array(pcd.points)
        
        select_m = np.dot(pointcloud_w,plane_model) + float(plane_model_ori[3])
        index_inliers = np.argwhere((select_m >=-0.001)).reshape(-1).astype(int)
        inliers = pointcloud_w[index_inliers]
        select_m = np.dot(inliers,plane_model) + float(plane_model_ori[3])
        og_boundary = (float(self.cfg.og_resolution.tabletop[0])/200.0,float(self.cfg.og_resolution.tabletop[1])/200.0)
        index_inliers = np.argwhere((select_m <=0.3)).reshape(-1).astype(int)
        inliers = inliers[index_inliers]
        ############### expanded og
        
        ####################
        whole_point_cloud = inliers.copy()
        index_inliers = np.argwhere((inliers[:,1]>=-og_boundary[1]+cam_pos[1])).reshape(-1).astype(int)
        inliers = inliers[index_inliers]
        index_inliers = np.argwhere((inliers[:,1]<=og_boundary[1]+cam_pos[1])).reshape(-1).astype(int)
        inliers = inliers[index_inliers]
        index_inliers = np.argwhere((inliers[:,0]>=-og_boundary[0]+cam_pos[0])).reshape(-1).astype(int)
        inliers = inliers[index_inliers]
        index_inliers = np.argwhere((inliers[:,0]<=og_boundary[0]+cam_pos[0])).reshape(-1).astype(int)
        inliers = inliers[index_inliers]
        objects_pcd = o3d.geometry.PointCloud()
        objects_pcd.points = o3d.utility.Vector3dVector(inliers)
        pts_tab = np.array(inliers)
        ###################################### TSDF
        # Define grid parameters
        # x_min, x_max = -1, 1
        # y_min, y_max = -1, 1
        # z_min, z_max = -1, 1
        # step = 0.1

        # # Generate grid points
        # x = np.arange(x_min, x_max + step, step)
        # y = np.arange(y_min, y_max + step, step)
        # z = np.arange(z_min, z_max + step, step)
        # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        # points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # # Create a mesh
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # mesh_grid = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     o3d.utility.Vector3dVector(points),
        #     o3d.utility.DoubleVector([step, step * 2])
        # )

        # # Visualize the mesh grid
        # o3d.visualization.draw_geometries([mesh, mesh_grid])
        # sdf = o3d.geometry.PointCloud.compute_point_cloud_distance(objects_pcd)

        # # Visualize the SDF
        # o3d.visualization.draw_geometries([sdf])
        # voxel_size = 0.01  # Voxel size in meters
        # volume_dimensions = (50, 50, 50)
        # tsdf_volume = cv2.kinFu_largeScale_TSDFVoxelSet.create(voxel_size)
        # tsdf_volume = np.zeros(volume_dimensions)
        # for point in objects_pcd.points:
        # # Convert point coordinates to voxel indices
        #     voxel_indices = (point / voxel_size).astype(int)
        #     # print(voxel_indices)
        #     voxel_indices[0] += 25
        #     voxel_indices[1] +=25
        #     voxel_indices = tuple(voxel_indices)
        #     # Update TSDF value at the corresponding voxel
        #     tsdf_volume[voxel_indices] = 1.0 
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Extract voxel coordinates with non-zero TSDF values
        # indices = np.nonzero(tsdf_volume)
        # ax.scatter(indices[0] * voxel_size, indices[1] * voxel_size, indices[2] * voxel_size, c='b', marker='.')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()
        # # Integrate point cloud into TSDF volume
        # for point in objects_pcd:
        #     tsdf_volume.integrate(point)

        # # Extract surface (optional)
        # vertices, triangles = tsdf_volume.getMesh()

        # # Visualize TSDF volume as a 3D point cloud
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        # # Visualize TSDF volume as a surface mesh
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for triangle in triangles:
        #     v0 = vertices[triangle[0]]
        #     v1 = vertices[triangle[1]]
        #     v2 = vertices[triangle[2]]
        #     triangle = np.array([v0, v1, v2, v0])
        #     ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'b-')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()



        ######################################
        # o3d.visualization.draw_geometries([objects_pcd])
        # print(camera.data.output["distance_to_image_plane"].shape)
        # print(pointcloud_w.shape)
        select_m = np.dot(inliers,plane_model) + float(plane_model_ori[3])
        index_objects = np.argwhere((select_m>=0.005)).reshape(-1).astype(int)
        pts = inliers[index_objects].copy()
        select_m = np.dot(pts,plane_model) + float(plane_model_ori[3])
        index_objects = np.argwhere((select_m<=0.1)).reshape(-1).astype(int)
        pts = pts[index_objects].copy()
        Nx = self.cfg.og_resolution.tabletop[0]
        Ny = self.cfg.og_resolution.tabletop[1]
        ###################### get the point cloud of objs
        point_obj = o3d.geometry.PointCloud()
        pts_tmp = pts.copy()
        pts_tmp[:,2] = 0
        point_obj.points = o3d.utility.Vector3dVector(pts_tmp)
        point_objs_1 = o3d.geometry.PointCloud()
        point_objs_1.points = o3d.utility.Vector3dVector(pts)
        # o3d.visualization.draw_geometries([point_objs_1])
        # o3d.visualization.draw_geometries([point_obj])
        x = np.linspace(np.min(pts_tab[:,0]), np.max(pts_tab[:,0]), Nx)
        y = np.linspace(np.min(pts_tab[:,1]), np.max(pts_tab[:,1]), Ny)
        xv, yv = np.meshgrid(x, y)

        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        # o3d.visualization.draw_geometries([pts_grid])
        distance = pts_grid.compute_point_cloud_distance(point_obj)
        # distance_2 = pts_grid.compute_point_cloud_distance(point_objs_1)
        # dist_2 = np.array(distance_2)
        # colormaps = cm.get_cmap('viridis')
        # colors = colormaps(dist_2.flatten())
        # print(colors)
        # pts_grid.colors = o3d.utility.Vector3dVector((colors[:,:3]*255).astype(np.uint8))
        # o3d.visualization.draw_geometries([point_objs_1])
        # o3d.visualization.draw_geometries([point_obj])
        dist = np.array(distance)
        # norm = mplc.Normalize(vmin=min(distance), vmax=max(distance), clip=True)
        Tsdf = dist.reshape(Ny,Nx)
        # Tsdf = np.rot90(Tsdf)
        Tsdf = np.fliplr(Tsdf)
        # plt.imshow(Tsdf)
        # plt.show()
        # Tsfd_t = Tsdf.copy()
        # idx_x = int(np.argmax(Tsfd_t)//50)
        # idx_y = int(np.argmax(Tsfd_t)%50)
        # Tsfd_t[max(idx_x-1,0):min(idx_x+2,50),max(idx_y-1,0):min(idx_y+2,50)] = 0.3
        # Tsfd_t[idx_x,idx_y] = 0.4
        # plt.imshow(Tsfd_t)
        # plt.show()
        ######################
        # print("tab size")
        # print(np.min(pts_tab[:,0]),np.min(pts_tab[:,1]),np.max(pts_tab[:,0]),np.max(pts_tab[:,1]))
        u = (pts[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
        v = (pts[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
        u = (Nx-1)*u
        v = (Ny-1)*v
        occupancy = np.zeros( (Ny,Nx) )
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
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
        # occupancy = np.rot90(occupancy)
        occupancy = np.fliplr(occupancy)
        # plt.imshow(occupancy)
        # plt.show()
        # if self.reset_f and self.show_first_og:
        #     # plt.imshow(occupancy)
        #     # plt.show()
        #     # self.reset_f = False
        #     self.show_first_og = False
        
        ################## expanded og
        
        index_inliers = np.argwhere((whole_point_cloud[:,1]>=-og_boundary[1]-0.06+cam_pos[1])).reshape(-1).astype(int)
        whole_point_cloud = whole_point_cloud[index_inliers]
        index_inliers = np.argwhere((whole_point_cloud[:,1]<=og_boundary[1]+0.06+cam_pos[1])).reshape(-1).astype(int)
        whole_point_cloud = whole_point_cloud[index_inliers]
        index_inliers = np.argwhere((whole_point_cloud[:,0]>=-og_boundary[0]-0.06+cam_pos[0])).reshape(-1).astype(int)
        whole_point_cloud = whole_point_cloud[index_inliers]
        index_inliers = np.argwhere((whole_point_cloud[:,0]<=og_boundary[0]+0.06+cam_pos[0])).reshape(-1).astype(int)
        whole_point_cloud = whole_point_cloud[index_inliers]
        select_m = np.dot(whole_point_cloud,plane_model) + float(plane_model_ori[3])
        index_objects = np.argwhere((select_m>=0.005)).reshape(-1).astype(int)
        pts_ex = whole_point_cloud[index_objects].copy()
        select_m = np.dot(pts_ex,plane_model) + float(plane_model_ori[3])
        index_objects = np.argwhere((select_m<=0.1)).reshape(-1).astype(int)
        pts_ex = pts_ex[index_objects].copy()
        Nx = self.cfg.og_resolution.tabletop[0]+int(self.cfg.og_resolution.ex_occu[0])*2
        Ny = self.cfg.og_resolution.tabletop[1]+int(self.cfg.og_resolution.ex_occu[0])*2
        # objects_pcd = o3d.geometry.PointCloud()
        # objects_pcd.points = o3d.utility.Vector3dVector(whole_point_cloud)
        # o3d.visualization.draw_geometries([objects_pcd])
        ################ original table
        shift_ind = int(self.cfg.og_resolution.ex_occu[0])
        # print(shift_ind)
        # print(Nx,Ny)
        occupancy_ex = np.zeros( (Ny,Nx) )
        occupancy_ex[shift_ind:self.cfg.og_resolution.tabletop[1]+shift_ind,shift_ind:self.cfg.og_resolution.tabletop[0]+shift_ind] = 0
        u = (pts_ex[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
        v = (pts_ex[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
        u = (Nx-shift_ind*2-1)*u +shift_ind
        v = (Ny-shift_ind*2-1)*v +shift_ind
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
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
        occupancy_ex[v,u] = 1
        # occupancy_ex = np.rot90(occupancy_ex)
        occupancy_ex = np.fliplr(occupancy_ex)
        # plt.imshow(occupancy_ex)
        # plt.show()
        kernel1 = np.ones((3, 3), np.float32)/9
        occupancy_ex_blur = occupancy_ex.copy()
        occupancy_ex_blur = cv2.filter2D(src=occupancy_ex, ddepth=-1, kernel=kernel1) 
        occupancy_blur = cv2.filter2D(src=occupancy, ddepth=-1, kernel=kernel1) 
        occupancy_blur[np.where(occupancy_blur>0.2)] = 1
        occupancy_ex_blur[np.where(occupancy_ex_blur>0.2)] = 1
        occupancy_blur[np.where(occupancy_blur<=0.2)] = 0
        # plt.imshow(occupancy_blur)
        # plt.show()
        # occupancy_ex_blur[np.where(occupancy_ex_blur<=0.2)] = 0
        # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
        # ax1.imshow(occupancy)
        # ax2.imshow(occupancy_blur)
        # plt.show()
        occupancy_ex_ori = occupancy_ex.copy()
        occupancy_ex = np.zeros( (Ny,Nx) )
        occupancy_ex[shift_ind:self.cfg.og_resolution.tabletop[1]+shift_ind,shift_ind:self.cfg.og_resolution.tabletop[0]+shift_ind] = 1
        occupancy_ex[np.where(occupancy_ex_blur>=0.5)] = 2
        tsdf_copy = Tsdf.copy()
        occupancy_blur_copy = occupancy_blur.copy()
        # plt.imshow(occupancy_ex)
        # plt.show()
        occu_t = occupancy_ex.copy()
        occu_t[shift_ind:self.cfg.og_resolution.tabletop[1]+shift_ind,shift_ind:self.cfg.og_resolution.tabletop[0]+shift_ind] = 1
        # plt.imshow(occu_t)
        # plt.show()
        occu_t = occupancy_ex.copy()
        occu_t[shift_ind-3:self.cfg.og_resolution.tabletop[1]+shift_ind+3,shift_ind-3:self.cfg.og_resolution.tabletop[0]+shift_ind+3] = 3
        occu_t[shift_ind:self.cfg.og_resolution.tabletop[1]+shift_ind,shift_ind:self.cfg.og_resolution.tabletop[0]+shift_ind] = 1
        # plt.imshow(occu_t)
        # plt.show()
        # for i in range(4):
        #     if i >0:
        #         occupancy_blur_copy= np.rot90(occupancy_blur_copy,k=-1)
        #         tsdf_copy = np.rot90(tsdf_copy,k=-1)
        #     fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
        #     ax1.imshow(occupancy_blur_copy)
        #     ax2.imshow(tsdf_copy)
        #     plt.show()
        return occupancy_blur,occupancy_ex,Tsdf
    def noise_tsdf(self,occu):
        point_obj = o3d.geometry.PointCloud()
        Nx,Ny = [50,50]
        if np.max(occu) >=0.5:
            occupied_ind = np.where(occu>=0.5)
            # print(occupied_ind)
            point_obj_points = np.zeros((len(occupied_ind[0]),3))
            point_obj_points[:,0] = occupied_ind[1].copy() * 0.01+0.005
            point_obj_points[:,1] = occupied_ind[0].copy() * 0.01+0.005
            # print(point_obj_points)
            point_obj.points = o3d.utility.Vector3dVector(point_obj_points)
            x = np.linspace(0,0.5, 50)
            y = np.linspace(0,0.5, 50)
            xv, yv = np.meshgrid(x, y)

            grid = np.zeros((Nx*Ny,3))
            grid[:,0] = xv.flatten()
            grid[:,1] = yv.flatten()
            pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
            distance = pts_grid.compute_point_cloud_distance(point_obj)
            dist = np.array(distance)
            # norm = mplc.Normalize(vmin=min(distance), vmax=max(distance), clip=True)
            Tsdf = dist.reshape(Ny,Nx)
            return Tsdf
        else:
            return None
    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        # apply to instance manager
        # -- goal
        # self._goal_markers.set_world_poses(self.object_des_pose_w[:, 0:3], self.object_des_pose_w[:, 3:7])
        # -- end-effector
        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
        # -- task-space commands
        if self.cfg.control.control_type == "inverse_kinematics":
            # convert to world frame
            ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
            ee_orientations = self._ik_controller.desired_ee_rot
            # set poses
            self._cmd_markers.set_world_poses(ee_positions, ee_orientations)

    """
    Helper functions - MDP.
    """
    def _get_obj_info(self,env_ids: VecEnvIndices):
        if len(self.new_obj_vertices)==0:
            self.new_obj_vertices = [i for i in env_ids.tolist()]
        for i in env_ids.tolist():
            mask_obj = self.obj_masks[i].cpu().numpy()
            obj_vertices = get_new_obj_contour_bbox(mask_obj) 
            self.new_obj_vertices[i] = obj_vertices
            
            # plt.imshow(mask_obj)
            # plt.show()
            # print(self.new_obj_vertices)
    def _place_obj(self,pos,rot,env_ids:VecEnvIndices,obj_type:str):
        for k in env_ids.tolist():
            if k in self.obj_on_table:
                env_ids_tmp = torch.from_numpy(np.array([k])).to(self.device)
                _ = self.obj_on_table_name[k][obj_type]
                if obj_type == self.cfg.YCBdata.ycb_name[0]:
                    root_state = self.obj1[_].get_default_root_state(env_ids_tmp)
                    root_state[:,0:3] = torch.from_numpy(np.array(pos)).to(self.device)
                    root_state[:, 3:7] = torch.from_numpy(np.array(rot)).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids_tmp]
                    self.obj1[_].set_root_state(root_state, env_ids=env_ids_tmp)
                    self.obj_on_table[k].append(self.obj1[_])
                elif obj_type == self.cfg.YCBdata.ycb_name[1]:
                    root_state = self.obj2[_].get_default_root_state(env_ids_tmp)
                    root_state[:,0:3] = torch.from_numpy(np.array(pos)).to(self.device)
                    root_state[:, 3:7] = torch.from_numpy(np.array(rot)).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids_tmp]
                    self.obj2[_].set_root_state(root_state, env_ids=env_ids_tmp)
                    self.obj_on_table[k].append(self.obj2[_])
                elif obj_type == self.cfg.YCBdata.ycb_name[2]:
                    root_state = self.obj3[_].get_default_root_state(env_ids_tmp)
                    root_state[:,0:3] = torch.from_numpy(np.array(pos)).to(self.device)
                    root_state[:, 3:7] = torch.from_numpy(np.array(rot)).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids_tmp]
                    self.obj3[_].set_root_state(root_state, env_ids=env_ids_tmp)
                    self.obj_on_table[k].append(self.obj3[_])
                elif obj_type == self.cfg.YCBdata.ycb_name[3]:
                    root_state = self.obj4[_].get_default_root_state(env_ids_tmp)
                    root_state[:,0:3] = torch.from_numpy(np.array(pos)).to(self.device)
                    root_state[:, 3:7] = torch.from_numpy(np.array(rot)).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids_tmp]
                    self.obj4[_].set_root_state(root_state, env_ids=env_ids_tmp)
                    self.obj_on_table[k].append(self.obj4[_])
                elif obj_type == self.cfg.YCBdata.ycb_name[4]:
                    root_state = self.obj5[_].get_default_root_state(env_ids_tmp)
                    root_state[:,0:3] = torch.from_numpy(np.array(pos)).to(self.device)
                    root_state[:, 3:7] = torch.from_numpy(np.array(rot)).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids_tmp]
                    self.obj5[_].set_root_state(root_state, env_ids=env_ids_tmp)
                    self.obj_on_table[k].append(self.obj5[_])
            self.obj_on_table_name[k][obj_type] +=1
        for _ in range(30):
            self.sim.step()
        self.pushing_policy_result[self.env_i_tmp][0] += 1
        self.pushing_policy_result[self.env_i_tmp][6] = self.pushing_policy_result[self.env_i_tmp][2]
        # print(self.pushing_policy_result)
        pass
    def _check_fallen_objs(self,env_ids:VecEnvIndices):
        torch_fallen = torch.zeros((self.num_envs,),device=self.device)
        for k in env_ids.tolist():
            if k in self.obj_on_table:
                for _,obj in enumerate(self.obj_on_table[k]):
                # print(obj.data.root_pos_w[1, :3])
                # if obj != 0:
                    torch_fallen[k] += torch.where(obj.data.root_pos_w[k, 2] < -0.05, 1, 0)
                    if obj.data.root_pos_w[k, 2] >= -0.05:
                        if obj.data.root_pos_w[k, 1]>=0.233:
                            torch_fallen[k] += 1
                        elif obj.data.root_pos_w[k, 1]<=-0.233:
                            torch_fallen[k] += 1
                        elif obj.data.root_pos_w[k, 0]<=-0.233:
                            torch_fallen[k] += 1
                        elif obj.data.root_pos_w[k, 0]>=0.233:
                            torch_fallen[k] += 1
        # print(torch_fallen)
        # print(self.falling_obj_all)
        self.falling_obj[env_ids] = torch_fallen[env_ids] - self.falling_obj_all[env_ids]
        self.pushing_policy_result[self.env_i_tmp][1] += int(self.falling_obj[0])
        self.pushing_policy_result[self.env_i_tmp][0] -= int(self.falling_obj[0])
        self.falling_obj_all[env_ids] = torch_fallen[env_ids]
        self.fallen_all += float(torch.sum(self.falling_obj))
        # print('fallen')
        # print(self.fallen_all)
        
    def _update_table_og(self):
        if self.reset_f:
            for i in range(self.num_envs):
                self.cams[i].update(self.dt)
                
                og,og_ex,tsdf = self.get_og(self.cams[i])
                # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
                # ax1.imshow(og)
                # ax2.imshow(tsdf)
                # plt.show()
                self.table_og[i] = torch.from_numpy(og.copy()).to(self.device)
                self.table_expand_og[i] = torch.from_numpy(og_ex.copy()).to(self.device)
                self.table_tsdf[i] = torch.from_numpy(tsdf.copy()).to(self.device)
                self.pushing_policy_result[self.env_i_tmp][3] = self.table_expand_og[i].clone().detach().cpu().numpy()

    def _check_placing(self):
        
        env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
        self._update_table_og()
        # for i in range(self.num_envs):
        #     plt.imshow()
        # print('check placing')
        # print(self.place_success_all)
        for i in range(self.num_envs):
            occupancy = self.table_og[i].cpu().numpy()
            vertices_new_obj = self.new_obj_vertices[i]
            # print('vertices',vertices_new_obj)
            flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(occupancy,vertices_new_obj)  
            # plt.imshow(occu_tmp)
            # plt.show()
            # flag_found = False
            if flag_found:
                self.place_success[i] = 1
                pos = [(self.cfg.og_resolution.tabletop[0]/2-new_obj_pos[1])*1./100.,(new_obj_pos[0]-self.cfg.og_resolution.tabletop[1]/2)*1./100.,0.05]
                obj_name_i=self.new_obj_type[i]
                # rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                if obj_name_i in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,np.rad2deg(new_obj_pos[2]),0), degrees=True).as_quat(), to="wxyz")
                else:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                
                env_ids_tmp = torch.from_numpy(np.array([i])).to(self.device)
                self._place_obj(pos,rot,env_ids_tmp,obj_name_i)
                for i in env_ids.tolist():
                    if len(self.obj_on_table[i])<len(self.obj_list):
                        self.new_obj_mask[i] = self.cfg.obj_mask.mask[self.cfg.YCBdata.ycb_name[self.obj_list[len(self.obj_on_table[i])]]]
                        self.new_obj_type[int(i)] = self.cfg.YCBdata.ycb_name[self.obj_list[len(self.obj_on_table[i])]] 
                        # print(self.new_obj_type[int(i)])
                        # self.pushing_policy_result[self.env_i_tmp][0] = len(self.obj_on_table[i])
                        self._get_obj_mask(env_ids=env_ids_tmp)
                        self._get_obj_info(env_ids=env_ids_tmp)
                        # plt.imshow(self.new_obj_mask[i])
                        # plt.show()
                    else:
                        self.stop_pushing[i] = 1
                        # print('stop')
                # print('place_success',env_ids_tmp)
                # plt.imshow(self.obj_masks[i].cpu().numpy())
                # plt.show()
                ############## visulaize placing
                self.place_success_all +=1.0
                return True
            else:
                    flag_found,new_obj_pos = placing_compare_fun_2(occupancy,self.obj_masks[i].clone().cpu().numpy())
                    if flag_found:
                        self.place_success[i] = 1
                        self.place_success_all +=1.0
                        pos = [(self.cfg.og_resolution.tabletop[0]/2-new_obj_pos[1])*1./100.,(new_obj_pos[0]-self.cfg.og_resolution.tabletop[1]/2)*1./100.,0.05]
                        obj_name_i=self.new_obj_type[i]
                        # rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                        if obj_name_i in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                            rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,np.rad2deg(new_obj_pos[2]),0), degrees=True).as_quat(), to="wxyz")
                        else:
                            rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,-np.rad2deg(new_obj_pos[2])), degrees=True).as_quat(), to="wxyz")
                        
                        env_ids_tmp = torch.from_numpy(np.array([i])).to(self.device)
                        self._place_obj(pos,rot,env_ids_tmp,obj_name_i)
                        for i in env_ids.tolist():
                            if len(self.obj_on_table[i])<len(self.obj_list):
                                self.new_obj_mask[i] = self.cfg.obj_mask.mask[self.cfg.YCBdata.ycb_name[self.obj_list[len(self.obj_on_table[i])]]]
                                self.new_obj_type[int(i)] = self.cfg.YCBdata.ycb_name[self.obj_list[len(self.obj_on_table[i])]] 
                                # print(self.new_obj_type[int(i)])
                                # self.pushing_policy_result[self.env_i_tmp][0] = len(self.obj_on_table[i])
                                self._get_obj_mask(env_ids=env_ids_tmp)
                                self._get_obj_info(env_ids=env_ids_tmp)
                            else:
                                self.stop_pushing[i] = 1
                                # print('stop')
                        return True
                # print('place')
                # print(env_ids_tmp)
                # self.un_satisfied_scenes.append(self.table_scenes[i])
                # print(self.place_success_all)
                # print(self.un_satisfied_scenes)
                # print('steps') 
                # print(self.step_all)
        # self.check_placing = False
        return False
    '''
    only for toy example
    '''            
    def _check_reaching_toy_v1(self):
        #print('check reaching')
        #print(self.actions_origin)
        self.check_reaching = torch.zeros((self.num_envs,),device=self.device)
        actions_tmp = self.actions_origin.clone().cpu().numpy().astype(np.uint8)
        patches_tmp = self.random_patches.clone().cpu().numpy()
        for i in range(self.num_envs):
            ind_x = actions_tmp[i][0]
            ind_y = actions_tmp[i][1]
            if patches_tmp[i,ind_x,ind_y] == 1:  
                self.check_reaching[i] = 1 
                
    '''for toy example v2'''     
    def _check_reaching_toy_v2(self):
        #print('check reaching')
        #print(self.actions_origin)
        self.out_side = torch.zeros((self.num_envs,),device=self.device)
        self.check_reaching = torch.zeros((self.num_envs,),device=self.device)
        # action_tmp_reward = self.actions.clone().cpu().numpy().astype(np.uint8)
        start_t = int((self.cfg.og_resolution.expand_obs[0]-self.cfg.og_resolution.tabletop[0])/2)
        for i in range(self.num_envs):
            table_og_tmp = self.table_og_pre[i].clone().cpu().numpy()
            ''' original reaching toy v2

            start_ind_x = max(self.actions_origin[i][0]-1,0)
            end_ind_x = min(self.actions_origin[i][0]+2,self.cfg.og_resolution.tabletop[0])
            start_ind_y = max(self.actions_origin[i][1]-5,0)
            end_ind_y = min(self.actions_origin[i][1],self.cfg.og_resolution.tabletop[1])
            if torch.sum(table_og_tmp[self.actions_origin[i][0],start_ind_y:end_ind_y])>=1:
                start_ind_y = max(self.actions_origin[i][1]-1,0)
                end_ind_y = min(self.actions_origin[i][1]+1,self.cfg.og_resolution.tabletop[1])
                if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])==0:
                    self.check_reaching[i] = 1   
            '''
            # print('checking start points')
            # plt.imshow(table_og_tmp.cpu().numpy())
            # plt.show()
            if (self.actions_origin[i][0] >=start_t and 
                self.actions_origin[i][0]<self.cfg.og_resolution.tabletop[0]+start_t and 
                self.actions_origin[i][1]>=start_t and 
                self.actions_origin[i][1]<self.cfg.og_resolution.tabletop[0]+start_t):
                self.actions_origin[i][0] -=start_t
                self.actions_origin[i][1] -=start_t
                # print(i,self.actions_origin[i])
                table_og_tmp1 = table_og_tmp.copy()
                if (self.actions_origin[i,2]%2) == 0:
                    if (self.actions_origin[i,2]%4) == 0:
                        start_ind_x = max(self.actions_origin[i][0]-1,0)
                        end_ind_x = min(self.actions_origin[i][0]+2,self.cfg.og_resolution.tabletop[0])
                        start_ind_y = max(self.actions_origin[i][1]-1,0)
                        end_ind_y = min(self.actions_origin[i][1]+2,self.cfg.og_resolution.tabletop[1])
                    else:
                        start_ind_x = max(self.actions_origin[i][0]-1,0)
                        end_ind_x = min(self.actions_origin[i][0]+2,self.cfg.og_resolution.tabletop[0])
                        start_ind_y = max(self.actions_origin[i][1]-1,0)
                        end_ind_y = min(self.actions_origin[i][1]+2,self.cfg.og_resolution.tabletop[1])
                    # print(start_ind_x,end_ind_x,start_ind_y,end_ind_y)
                    # table_og_tmp1[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 3
                    # table_og_tmp1[min(end_ind_x,49),min(end_ind_y,49)] = 2
                    if np.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])==0:
                        if self.actions_origin[i,2] == 0:
                            start_ind_y = max(self.actions_origin[i][1]-5,0)
                            end_ind_y = min(self.actions_origin[i][1],self.cfg.og_resolution.tabletop[1])
                            if np.sum(table_og_tmp[self.actions_origin[i][0],start_ind_y:end_ind_y])>=1:
                                self.check_reaching[i] = 1 
                        elif self.actions_origin[i,2] == 2:
                            start_ind_x = max(self.actions_origin[i][0],0)
                            end_ind_x = min(self.actions_origin[i][0]+5,self.cfg.og_resolution.tabletop[0])
                            if np.sum(table_og_tmp[start_ind_x:end_ind_x,self.actions_origin[i][1]])>=1:
                                self.check_reaching[i] = 1 
                        elif self.actions_origin[i,2] == 4:
                            start_ind_y = max(self.actions_origin[i][1],0)
                            end_ind_y = min(self.actions_origin[i][1]+5,self.cfg.og_resolution.tabletop[1])
                            if np.sum(table_og_tmp[self.actions_origin[i][0],start_ind_y:end_ind_y])>=1:
                                self.check_reaching[i] = 1 
                        elif self.actions_origin[i,2] == 6:
                            start_ind_x = max(self.actions_origin[i][0]-5,0)
                            end_ind_x = min(self.actions_origin[i][0],self.cfg.og_resolution.tabletop[0])
                            if np.sum(table_og_tmp[start_ind_x:end_ind_x,self.actions_origin[i][1]])>=1:
                                self.check_reaching[i] = 1  
                    # table_og_tmp1[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 2
                    # print(self.actions_origin[i])
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7, 4))
                    # ax1.imshow(table_og_tmp1.cpu().numpy())
                    # ax2.imshow(table_og_tmp.cpu().numpy())
                    # plt.show()
                    
                    del table_og_tmp,table_og_tmp1
                else:
                    
                    start_ind_x = max(self.actions_origin[i][0]-1,0)
                    end_ind_x = min(self.actions_origin[i][0]+2,self.cfg.og_resolution.tabletop[0])
                    start_ind_y = max(self.actions_origin[i][1]-1,0)
                    end_ind_y = min(self.actions_origin[i][1]+2,self.cfg.og_resolution.tabletop[1])
                    touch_item = False
                    # table_og_tmp1[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 2
                    # print('start',torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y]))
                    if np.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])==0:
                        if self.actions_origin[i,2] == 1:
                            
                            for j in range(4):
                                x_tmp = self.actions_origin[i][0] +j
                                y_tmp = self.actions_origin[i][1] -j
                                if x_tmp >=self.cfg.og_resolution.tabletop[0] or y_tmp<0 or touch_item:
                                    break
                                if table_og_tmp[x_tmp,y_tmp] >0:
                                   table_og_tmp1[x_tmp,y_tmp] = 2
                                   touch_item = True
                        elif self.actions_origin[i,2] == 3:
                            
                            for j in range(4):
                                x_tmp = self.actions_origin[i][0] +j
                                y_tmp = self.actions_origin[i][1] +j
                                if x_tmp >=self.cfg.og_resolution.tabletop[0] or y_tmp>=self.cfg.og_resolution.tabletop[0] or touch_item:
                                    break
                                if table_og_tmp[x_tmp,y_tmp] >0:
                                   table_og_tmp1[x_tmp,y_tmp] = 2
                                   touch_item = True
                        elif self.actions_origin[i,2] == 5:
                            
                            for j in range(4):
                                x_tmp = self.actions_origin[i][0] -j
                                y_tmp = self.actions_origin[i][1] +j
                                if x_tmp <0 or y_tmp>=self.cfg.og_resolution.tabletop[0] or touch_item:
                                    break
                                if table_og_tmp[x_tmp,y_tmp] >0:
                                   table_og_tmp1[x_tmp,y_tmp] = 2
                                   touch_item = True        
                        elif self.actions_origin[i,2] == 7:
                            
                            for j in range(4):
                                x_tmp = self.actions_origin[i][0] -j
                                y_tmp = self.actions_origin[i][1] -j
                                if x_tmp <0 or y_tmp<0 or touch_item:
                                    break
                                if table_og_tmp[x_tmp,y_tmp] >0:
                                   table_og_tmp1[x_tmp,y_tmp] = 2
                                   touch_item = True 
                    if touch_item:
                        self.check_reaching[i] = 1
                    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7, 4))
                    # ax1.imshow(table_og_tmp1.cpu().numpy())
                    # ax2.imshow(table_og_tmp.cpu().numpy())
                    # plt.show()
                    del table_og_tmp,table_og_tmp1

    def _check_termination(self) -> None:
        # access buffers from simulator
        # object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when stop pushing
        if self.cfg.terminations.stop_pushing:
            # if not self.cfg.flag_compare:
            #     self.reset_buf = torch.where(self.check_reaching == 0, 1, self.reset_buf)
            self.reset_buf = torch.where(self.stop_pushing >=0.5,1,self.reset_buf)
        # -- when task is successful
        if self.cfg.terminations.is_success:
            ''' modified because of toy example'''
            if self.cfg.pre_train:
                self.reset_buf = torch.where(self.check_reaching >= 0.5, 1, self.reset_buf)
            else:
                self.reset_buf = torch.where(self.place_success >= 0.5, 1, self.reset_buf)
            ''' modified because of toy example v2'''
            # self.reset_buf = torch.where(self.check_reaching >= 0.5, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
        # print('terminated',self.reset_buf)
    def reset_objs(self,env_ids: torch.Tensor):
        for i,obj_t in enumerate(self.obj1):
            root_state = obj_t.get_default_root_state(env_ids)
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # set the root state
            obj_t.set_root_state(root_state, env_ids=env_ids)
        for i,obj_t in enumerate(self.obj2):
            root_state = obj_t.get_default_root_state(env_ids)
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # set the root state
            obj_t.set_root_state(root_state, env_ids=env_ids)
        for i,obj_t in enumerate(self.obj3):
            root_state = obj_t.get_default_root_state(env_ids)
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # set the root state
            obj_t.set_root_state(root_state, env_ids=env_ids)
        for i,obj_t in enumerate(self.obj4):
            root_state = obj_t.get_default_root_state(env_ids)
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # set the root state
            obj_t.set_root_state(root_state, env_ids=env_ids)
        for i,obj_t in enumerate(self.obj5):
            root_state = obj_t.get_default_root_state(env_ids)
            # transform command from local env to world
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # set the root state
            obj_t.set_root_state(root_state, env_ids=env_ids)
        
    def _randomize_table_scene(self,env_ids: torch.Tensor):
        
        file_name = self.cfg.env_name
        # ycb_usd_paths = self.cfg.YCBdata.ycb_usd_paths
        ycb_name = self.cfg.YCBdata.ycb_name
        # self.obj_dict = dict()
        for i in env_ids.tolist():
            self.obj_on_table[i] = []
            self.obj_on_table_name[i] = dict()
            for j in self.cfg.YCBdata.ycb_name:
                self.obj_on_table_name[i][j] = 0
        # self.obj_on_table = []
        num_env = len(file_name)
        self.env_i_tmp +=1
        choosen_env_id = np.random.randint(0,num_env)
        # choosen_env_id = self.env_i_tmp
        # print(file_name[choosen_env_id],env_ids,self.env_i_tmp,choosen_env_id)
        # env_path = "generated_table2/"+file_name[choosen_env_id]
        f_name = 'dict_pose_'+str(self.env_i_tmp)+'.pkl'
        # f_name = 'dict_pose_'+str(52)+'.pkl'
        # print('env name',f_name)
        if self.env_i_tmp>50:
            file_path = "./placing_test/pushing_FCN_withoutmask_"+str(self.env_i_tmp-1)+".pkl"
            # file_path = "./placing_test/pushing_FCN_"+str(52)+".pkl"
            f_save = open(file_path,'wb')
            
            pkl.dump(self.pushing_policy_result[self.env_i_tmp-1],f_save)
            f_save.close()
        if f_name in file_name:
            env_path = "placing_test/"+f_name
            self.pushing_policy_result[self.env_i_tmp] = [0,0,0,0,0,0,0] ### item be on table, fallen item, pushing steps, occu_table, original number of item ## last last steps num for item placed
        else:
            file_path = "./placing_test/pushing_FCN_withoutmask.pkl"
            f_save = open(file_path,'wb')
            
            pkl.dump(self.pushing_policy_result,f_save)
            f_save.close()
            self.close()
        
            
        # print(env_path)
        # env_path = "generated_table2/dict_478.pkl"
        for i in env_ids.tolist():
            self.table_scenes[i] = file_name[choosen_env_id]
        
        # else:
        #     print('steps')
        #     print(self.step_all)
        #     print('place')
        #     print(self.place_success_all)
        #     print('reach')
        #     print(self.reaching_all)
        #     print('fallen')
        #     print(self.fallen_all)
        #     print(self.un_satisfied_scenes)
        #     file_path = "un_satisfied_scenes.pkl"
        #     f_save = open(file_path,'wb')
            
        #     pkl.dump(self.un_satisfied_scenes,f_save)
        #     f_save.close()
        #     self.close()
            
        fileObject2 = open(env_path, 'rb')
        env =  pkl.load(fileObject2)
        self.obj_list = env[0]
        obj_pos_rot = env[1]
        # print('new obj list', self.obj_list)
        # print('old pos',obj_pos_rot)
        # obj_pos_rot = []
        # print(env)
        # self.new_obj_mask = self.cfg.obj_mask.mask["tomatoSoupCan"
        ycb_name = ['sugarBox','mustardBottle','tomatoSoupCan','Cube3','Cube2']
        
        
            # print('get new mask')
            # plt.imshow(self.new_obj_mask[i])
            # plt.show()
            # print(self.new_obj_type)
        # self.new_obj_type = "tomatoSoupCan"
        fileObject2.close()
        # print(env_ids)
        # for i in self.obj_list:
        #     obj_pos_rot.append(ycb_name[i])
        for i in obj_pos_rot:
            if i == ycb_name[0]:
                for _,pos_rot in enumerate(obj_pos_rot[i]):
                    if _ > 8:
                        break
                    root_state = self.obj1[_].get_default_root_state(env_ids)
                    for j in range(len(root_state)):
                        root_state[j, 0:3] = torch.from_numpy(np.array(pos_rot[0])).to(self.device)
                        root_state[j, 3:7] = torch.from_numpy(np.array(pos_rot[1])).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids]
                    self.obj1[_].set_root_state(root_state, env_ids=env_ids)
                    for j in env_ids.tolist():
                        self.obj_on_table[j].append(self.obj1[_])
                        self.obj_on_table_name[j][i] +=1
            elif i == ycb_name[1]:
                for _,pos_rot in enumerate(obj_pos_rot[i]):
                    if _ > 8:
                        break
                    root_state = self.obj2[_].get_default_root_state(env_ids)
                    for j in range(len(root_state)):
                        root_state[j, 0:3] = torch.from_numpy(np.array(pos_rot[0])).to(self.device)
                        root_state[j, 3:7] = torch.from_numpy(np.array(pos_rot[1])).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids]
                    self.obj2[_].set_root_state(root_state, env_ids=env_ids)
                    for j in env_ids.tolist():
                        self.obj_on_table[j].append(self.obj2[_])
                        self.obj_on_table_name[j][i] +=1
            elif i == ycb_name[2]:
                for _,pos_rot in enumerate(obj_pos_rot[i]):
                    if _ > 8:
                        break
                    root_state = self.obj3[_].get_default_root_state(env_ids)
                    for j in range(len(root_state)):
                        root_state[j, 0:3] = torch.from_numpy(np.array(pos_rot[0])).to(self.device)
                        root_state[j, 3:7] = torch.from_numpy(np.array(pos_rot[1])).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids]
                    self.obj3[_].set_root_state(root_state, env_ids=env_ids)
                    for j in env_ids.tolist():
                        self.obj_on_table[j].append(self.obj3[_])
                        self.obj_on_table_name[j][i] +=1    
            elif i == ycb_name[3]:
                for _,pos_rot in enumerate(obj_pos_rot[i]):
                    if _ > 16:
                        break
                    root_state = self.obj4[_].get_default_root_state(env_ids)
                    for j in range(len(root_state)):
                        root_state[j, 0:3] = torch.from_numpy(np.array(pos_rot[0])).to(self.device)
                        root_state[j, 3:7] = torch.from_numpy(np.array(pos_rot[1])).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids]
                    self.obj4[_].set_root_state(root_state, env_ids=env_ids)
                    for j in env_ids.tolist():
                        self.obj_on_table[j].append(self.obj4[_])
                        self.obj_on_table_name[j][i] +=1     
            elif i == ycb_name[4]:
                for _,pos_rot in enumerate(obj_pos_rot[i]):
                    if _ > 16:
                        break
                    root_state = self.obj5[_].get_default_root_state(env_ids)
                    for j in range(len(root_state)):
                        root_state[j, 0:3] = torch.from_numpy(np.array(pos_rot[0])).to(self.device)
                        root_state[j, 3:7] = torch.from_numpy(np.array(pos_rot[1])).to(self.device)
                    root_state[:, 0:3] += self.envs_positions[env_ids]
                    self.obj5[_].set_root_state(root_state, env_ids=env_ids)
                    for j in env_ids.tolist():
                        self.obj_on_table[j].append(self.obj5[_])
                        self.obj_on_table_name[j][i] +=1     
        
        for i in env_ids.tolist():
            self.new_obj_mask[i] = self.cfg.obj_mask.mask[ycb_name[self.obj_list[len(self.obj_on_table[i])]]]
            self.new_obj_type[int(i)] = ycb_name[self.obj_list[len(self.obj_on_table[i])]] 
            self.pushing_policy_result[self.env_i_tmp][0] = int(len(self.obj_on_table[i]))
            self.pushing_policy_result[self.env_i_tmp][4] = int(len(self.obj_on_table[i]))
            # plt.imshow(self.new_obj_mask[i])
            # plt.show()
            # mask_t = self.new_obj_mask[i].copy()
            # mask_t_x = np.diff(mask_t,axis=0)
            # plt.imshow(mask_t_x)
            # plt.show()
            # mask_t_y = np.diff(mask_t,axis=1)
            # plt.imshow(mask_t_y)
            # plt.show()
            # mask_t[np.where(mask_t_x!=0)]= 2
            # mask_t[np.where(mask_t_y!=0)]= 2
            # mask_t[9,15] = 2
            # plt.imshow(mask_t)
            # plt.show()
    def _randomize_object_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectInitialPoseCfg):
        """Randomize the initial pose of the object."""
        
        # get the default root state
        root_state = self.object.get_default_root_state(env_ids)
        # -- object root position
        if cfg.position_cat == "default":
            pass
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            root_state[:, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
        # -- object root orientation
        if cfg.orientation_cat == "default":
            pass
        elif cfg.orientation_cat == "uniform":
            # sample uniformly in SO(3)
            root_state[:, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(f"Invalid category for randomizing the object orientation '{cfg.orientation_cat}'.")
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # update object init pose
        self.object_init_pose_w[env_ids] = root_state[:, 0:7]
        # set the root state
        self.object.set_root_state(root_state, env_ids=env_ids)

   
    def _randomize_object_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectDesiredPoseCfg):
        """Randomize the desired pose of the object."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.object_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.object_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.object_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.object_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.object_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]


class PushObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""
    ''' for beast env v6
    def table_scene(self,env:PushEnv):
        # print("get observs")
        # obs_ta = torch.zeros((env.num_envs,env.cfg.og_resolution.tabletop[1]+12,
        #                             env.cfg.og_resolution.tabletop[0]+12,1),device=env.device)
        obs_ta = torch.zeros((env.num_envs,env.cfg.og_resolution.tabletop[1],
                                    env.cfg.og_resolution.tabletop[0],1),device=env.device)
        for i in range(env.num_envs):
            
            # im = env.table_expand_og[i].cpu().numpy()*255/2.0
            im = env.table_og[i].cpu().numpy()*255
            # print('obs output')
            # plt.imshow(im)
            # plt.show()
            observation = np.array(im,dtype=np.uint8)
            # observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1]+12,
            #                         env.cfg.og_resolution.tabletop[0]+12])
            observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1],
                                    env.cfg.og_resolution.tabletop[0]])
            obs_ta[i,:,:,0] = torch.from_numpy(observation).to(env.device)
        return obs_ta
    '''
    '''for local v10 env add tsdf'''
    def table_scene(self,env:PushEnv):
        # print("get observs")
        # obs_ta = torch.zeros((env.num_envs,env.cfg.og_resolution.tabletop[1]+12,
        #                             env.cfg.og_resolution.tabletop[0]+12,1),device=env.device)
        obs_ta = torch.zeros((env.num_envs,env.cfg.og_resolution.expand_obs[1],
                                    env.cfg.og_resolution.expand_obs[0],2),device=env.device)
        for i in range(env.num_envs):
            
            im = env.table_expand_og[i].clone().cpu().numpy()*255/2.0
            # im = env.table_og[i].cpu().numpy()*255
            
            # print('obs output')
            # plt.imshow(im)
            # plt.show()
            start_t = int((env.cfg.og_resolution.expand_obs[1] -env.cfg.og_resolution.tabletop[1])/2)
            observation = np.array(im,dtype=np.uint8)
            # observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1]+12,
            #                         env.cfg.og_resolution.tabletop[0]+12])
            observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1]+12,
                                    env.cfg.og_resolution.tabletop[0]+12])
            obs_ta[i,start_t-6:env.cfg.og_resolution.expand_obs[1]-start_t+6,
                   start_t-6:env.cfg.og_resolution.expand_obs[1]-start_t+6,0] = torch.from_numpy(observation).to(env.device)

            im_tsdf = env.table_tsdf[i].clone().cpu().numpy()/float(np.max(env.table_tsdf[i].clone().cpu().numpy()))*255
            observation = np.array(im_tsdf,dtype=np.uint8)
            # observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1]+12,
            #                         env.cfg.og_resolution.tabletop[0]+12])
            observation = observation[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1],
                                    env.cfg.og_resolution.tabletop[0]])
            obs_ta[i,start_t:env.cfg.og_resolution.expand_obs[1]-start_t,
                   start_t:env.cfg.og_resolution.expand_obs[1]-start_t,1] = torch.from_numpy(observation).to(env.device)
            
            # obs_ta=obs_ta.rot90(1,[2,1])
            # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
            # ax1.imshow(np.squeeze(obs_ta[i,:,:,1].cpu().numpy()))
            # ax2.imshow(np.squeeze(obs_ta[i,:,:,0].cpu().numpy()))
            # plt.show()
        return obs_ta
        # return env.table_og
        # return env.table_og
    def obs_for_toy_example(self,env:PushEnv):
        # push_trunk = torchvision.models.densenet.densenet121(pretrained=True).cuda()
        obs_ta = torch.zeros((env.num_envs,env.cfg.og_resolution.tabletop[1],
                                    env.cfg.og_resolution.tabletop[0],1),device=env.device)
        # densenet_input_tmp = torch.zeros((env.num_envs,3,env.cfg.og_resolution.tabletop[1],
        #                             env.cfg.og_resolution.tabletop[0]),device=env.device)
        for i in range(env.num_envs):
            patch_tmp = env.random_patches[i].cpu().numpy()
            '''
            start_i_x = max(int(env.actions_origin[i,0].cpu()-1),0)
            end_i_x = min(int(env.actions_origin[i,0].cpu()+1),env.cfg.og_resolution.tabletop[0])
            start_i_y = max(int(env.actions_origin[i,1].cpu()-1),0)
            end_i_y = min(int(env.actions_origin[i,1].cpu()+1),env.cfg.og_resolution.tabletop[1])
            patch_tmp[start_i_x:end_i_x,start_i_y:end_i_y] = 2.0
            patch_tmp = patch_tmp/2.0*255.0'''
            patch_tmp = patch_tmp*255.0
            patch_tmp = np.array(patch_tmp,dtype=np.uint8)
            patch_tmp = patch_tmp[:,np.newaxis].reshape([env.cfg.og_resolution.tabletop[1],
                                    env.cfg.og_resolution.tabletop[0]])
            obs_ta[i,:,:,0] = torch.from_numpy(patch_tmp).to(env.device)
            
            # patch_tmp = patch_tmp[np.newaxis,:].reshape([env.cfg.og_resolution.tabletop[1],
            #                         env.cfg.og_resolution.tabletop[0]])
            # densenet_input_tmp[i,:] = torch.from_numpy(patch_tmp[0]).to(env.device)
        # interm_push_color_feat = push_trunk.features(densenet_input_tmp)
        # print(interm_push_color_feat)

        return obs_ta
    # def new_obj_mask(self,env:PushEnv):
    #     # print(env.new_obj_mask.shape)
        
    #     return obs_mask
        # return env.obj_masks
    def arm_dof_pos(self, env: PushEnv):
        """DOF positions for the arm."""
        return env.robot.data.arm_dof_pos

    def arm_dof_pos_scaled(self, env: PushEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 1],
        )

    def arm_dof_vel(self, env: PushEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def tool_dof_pos_scaled(self, env: PushEnv):
        """DOF positions of the tool normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.tool_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 1],
        )

    def tool_positions(self, env: PushEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def tool_orientations(self, env: PushEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    # def object_positions(self, env: PushEnv):
    #     """Current object position."""
    #     return env.object.data.root_pos_w - env.envs_positions

    # def object_orientations(self, env: PushEnv):
    #     """Current object orientation."""
    #     # make the first element positive
    #     quat_w = env.object.data.root_quat_w
    #     quat_w[quat_w[:, 0] < 0] *= -1
    #     return quat_w

    # def object_relative_tool_positions(self, env:PushEnv):
    #     """Current object position w.r.t. end-effector frame."""
    #     return env.object.data.root_pos_w - env.robot.data.ee_state_w[:, :3]

    # def object_relative_tool_orientations(self, env: PushEnv):
    #     """Current object orientation w.r.t. end-effector frame."""
    #     # compute the relative orientation
    #     quat_ee = quat_mul(quat_inv(env.robot.data.ee_state_w[:, 3:7]), env.object.data.root_quat_w)
    #     # make the first element positive
    #     quat_ee[quat_ee[:, 0] < 0] *= -1
    #     return quat_ee

    # def object_desired_positions(self, env: PushEnv):
    #     """Desired object position."""
    #     return env.object_des_pose_w[:, 0:3] - env.envs_positions

    # def object_desired_orientations(self, env: PushEnv):
    #     """Desired object orientation."""
    #     # make the first element positive
    #     quat_w = env.object_des_pose_w[:, 3:7]
    #     quat_w[quat_w[:, 0] < 0] *= -1
    #     return quat_w

    def arm_actions(self, env: PushEnv):
        """Last arm actions provided to env."""
        return env.actions[:, :-1]

    def tool_actions(self, env: PushEnv):
        """Last tool actions provided to env."""
        return env.actions[:, -1].unsqueeze(1)

    def tool_actions_bool(self, env: PushEnv):
        """Last tool actions transformed to a boolean command."""
        return torch.sign(env.actions[:, -1]).unsqueeze(1)


class PushRewardManager(RewardManager):
    """Reward manager for single-arm object lifting environment."""

    # def reaching_object_position_l2(self, env: PushEnv):
    #     """Penalize end-effector tracking position error using L2-kernel."""
    #     return torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3]), dim=1)

    # def reaching_object_position_exp(self, env: PushEnv, sigma: float):
    #     """Penalize end-effector tracking position error using exp-kernel."""
    #     error = torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3]), dim=1)
    #     return torch.exp(-error / sigma)

    # def reaching_object_position_tanh(self, env: PushEnv, sigma: float):
    #     """Penalize tool sites tracking position error using tanh-kernel."""
    #     # distance of end-effector to the object: (num_envs,)
    #     ee_distance = torch.norm(env.robot.data.ee_state_w[:, 0:3], dim=1)
    #     # distance of the tool sites to the object: (num_envs, num_tool_sites)
    #     # object_root_pos = env.object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    #     tool_sites_distance = torch.norm(env.robot.data.tool_sites_state_w[:, :, :3], dim=-1)
    #     # average distance of the tool sites to the object: (num_envs,)
    #     # note: we add the ee distance to the average to make sure that the ee is always closer to the object
    #     num_tool_sites = tool_sites_distance.shape[1]
    #     average_distance = (ee_distance + torch.sum(tool_sites_distance, dim=1)) / (num_tool_sites + 1)

    #     return 1 - torch.tanh(average_distance / sigma)

    # def penalizing_arm_dof_velocity_l2(self, env: PushEnv):
    #     """Penalize large movements of the robot arm."""
    #     return -torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    # def penalizing_tool_dof_velocity_l2(self, env: PushEnv):
    #     """Penalize large movements of the robot tool."""
    #     return -torch.sum(torch.square(env.robot.data.tool_dof_vel), dim=1)

    # def penalizing_arm_action_rate_l2(self, env: PushEnv):
    #     """Penalize large variations in action commands besides tool."""
    #     return -torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)
    def reward_max_tsdf_increase(self,env:PushEnv):
        max_tsdf_increase = torch.zeros((env.num_envs,),device=self.device)
        for i in range(env.num_envs):
            if env.check_reaching[i]>0.5:
                max_pre = torch.max(env.table_tsdf_pre[i])
                max_curr = torch.max(env.table_tsdf[i])
                if max_curr > max_pre and not env.cfg.pre_train:
                    if env.falling_obj[i]==0:
                        # max_tsdf_increase[i] = max_curr - max_pre
                        max_tsdf_increase[i] = 0.5
        # print('max_tsdf_increase')
        # print(max_tsdf_increase)
        return max_tsdf_increase
    def penaltizing_pushing_outside(self,env:PushEnv):
        pixel_outside_table = torch.zeros((env.num_envs,),device=self.device)
        env_tab_ex_tmp = env.table_expand_og.clone()
        env_tab_ex_tmp_pre = env.table_expand_og_pre.clone()
        for i in range(env.num_envs):
            # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(7, 4))
            # ax1.imshow(env_tab_ex_tmp[i].cpu().numpy())
            # ax2.imshow(env_tab_ex_tmp_pre[i].cpu().numpy())
            # plt.show()
            env_tab_ex_tmp[i][6:env.cfg.og_resolution.tabletop[1]+6,
                           6:env.cfg.og_resolution.tabletop[0]+6] = 0
            env_tab_ex_tmp_pre[i][6:env.cfg.og_resolution.tabletop[1]+6,
                           6:env.cfg.og_resolution.tabletop[0]+6] = 0
            # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(7, 4))
            # ax1.imshow(env_tab_ex_tmp[i].cpu().numpy())
            # ax2.imshow(env_tab_ex_tmp_pre[i].cpu().numpy())
            # plt.show()
            if not env.cfg.pre_train:
                if torch.sum(env_tab_ex_tmp[i])-torch.sum(env_tab_ex_tmp_pre[i])<=6:
                    if env.check_reaching[i]>0.5:
                        pixel_outside_table[i] = 0.0
                else:
                    if env.check_reaching[i]>0.5 and env.falling_obj[i]==0:
                        pixel_outside_table[i] = -2
                        env.push_out[i] = 1
                        # pixel_outside_table[i] = -2.0
            # else:

            # pixel_outside_table[i] = torch.sum(env_tab_ex_tmp[i])-torch.sum(env_tab_ex_tmp_pre[i])
               
            # if pixel_outside_table[i] >0:
            #     pixel_outside_table[i] = 0
            # else:
            #     pixel_outside_table[i]
        # print(-pixel_outside_table.type(torch.float16)/float(300.0))
        # return -pixel_outside_table.type(torch.float16)/float(20.0)
        # print('pushing outside')
        # print(pixel_outside_table.type(torch.float16))
        return pixel_outside_table.type(torch.float16)
    def penaltizing_repeat_actions(self,env:PushEnv):
        # print("repeat")
        # print(env.delta_same_action)
        return -env.delta_same_action.type(torch.float16) 
    def penaltizing_stop(self,env:PushEnv):
        # print('stop')
        # print(-env.stop_pushing.type(torch.float16))
        return -env.stop_pushing.type(torch.float16)
    def penaltizing_falling(self,env:PushEnv):
        # print("penalty fallen")
        # print(-env.falling_obj)
        # print("falling")
        # print(env.falling_obj)
        return -env.falling_obj.type(torch.float16)
    def check_placing(self,env:PushEnv):
        """try to place new object"""
        # print("reawrd success")
        # print(env.place_success)
        # print('placing')
        # print(env.place_success)
        # print(env.place_success)
        return env.place_success.type(torch.float16)
    def penaltizing_steps(self,env:PushEnv):
        # print("pernalize steps")
        # print(-torch.where(env.step_count!=0,1,0).to(env.device))
        # print('steps')
        # print(torch.where(env.step_count!=0,1,0).to(env.device))
        return -torch.where(env.step_count!=0,1,0).to(env.device).type(torch.float16)
    def reward_og_change(self,env:PushEnv):
        delta_og = torch.zeros((env.num_envs,),device=self.device)
        for i in range(env.num_envs):
            delta_tmp = env.table_og[i].clone() - env.table_og_pre[i].clone()
            if torch.sum(torch.abs(delta_tmp))>15 and env.check_reaching[i]>0.5 and env.falling_obj[i]==0:
                delta_og[i] = 0.5
                # if torch.sum(torch.abs(delta_tmp))<60:
                #     delta_og[i] = 1
                # else:
                #     delta_og[i] = 2
        # env.table_og_pre = env.table_og.clone()
        # print("reward og")
        # print(delta_og)
        # print('og change')
        # print(delta_og)
        return delta_og.type(torch.float16)
    def reward_near_obj_1(self,env:PushEnv):
        # print('reward 1')
        reward_near = torch.zeros((env.num_envs,),device=self.device)
        action_tmp_reward = env.actions.clone().cpu().numpy().astype(np.uint8)
        for i in range(env.num_envs):
            table_og_tmp = env.table_og_pre[i].clone()
            if env.actions_origin[i][1]>=3:
                if torch.sum(table_og_tmp[env.actions_origin[i][0],env.actions_origin[i][1]-3:env.actions_origin[i][1]])>=1:
                    reward_near[i] = 1
                    if env.actions_origin[i][0]<=env.cfg.og_resolution.tabletop[0]-2 and env.actions_origin[i][0]>=1:
                        if env.actions_origin[i][1]<=env.cfg.og_resolution.tabletop[1]-2:
                            # table_og_tmp = env.table_og[i].clone()
                            # print(table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2])
                            if torch.sum(table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2])==0:
                                reward_near[i] += 0.5
                            table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2]=2
                            table_og_tmp[env.actions_origin[i][0],env.actions_origin[i][1]-3:env.actions_origin[i][1]] = 2
                            # print('pushing position')
                            # plt.imshow(table_og_tmp.cpu().numpy())
                            # plt.show()
        return reward_near
    def reward_distribution_closer(self,env:PushEnv):
        delta_og = torch.zeros((env.num_envs,),device=self.device)
        for i in range(env.num_envs):
            ind_pre = torch.where(env.table_og_pre[i]>=0.8)
            ind_cur = torch.where(env.table_og[i]>=0.8)
            # ind_pre = ind_pre.cpu().numpy().reshape(-1,2)
            # ind_cur = ind_cur.cpu().numpy().reshape(-1,2)
            # print(ind_cur[0].shape)
            # print(ind_cur[1].shape)
            # plt.imshow(env.table_og_pre[i][ind_pre[0],ind_pre[1]].cpu().numpy())
            # plt.show()
            ind_pre_var_x = np.std(ind_pre[0].cpu().numpy())
            ind_pre_var_y = np.std(ind_pre[1].cpu().numpy())
            ind_cur_var_x = np.std(ind_cur[0].cpu().numpy())
            ind_cur_var_y = np.std(ind_cur[1].cpu().numpy())
            if ind_cur_var_x < ind_pre_var_x-0.1:
                delta_og[i] +=1.0
            if ind_cur_var_y < ind_pre_var_y-0.1:
                delta_og[i] +=1.0
            # if ind_cur_var_x > ind_pre_var_x+0.1:
            #     delta_og[i] -=0.2
            # if ind_cur_var_y > ind_pre_var_y+0.1:
            #     delta_og[i] -=0.2
            # print(ind_cur_var_x,ind_cur_var_y,ind_pre_var_x,ind_pre_var_y)
        
        # print(delta_og)
        # print("reward og")
        # print(delta_og)
        return delta_og.type(torch.float16)
    def reward_near_obj(self,env:PushEnv):
        reward_near = torch.zeros((env.num_envs,),device=self.device)
        action_tmp_reward = env.actions.clone().cpu().numpy().astype(np.uint8)
        for i in range(env.num_envs):
            table_og_tmp = env.table_og_pre[i].clone()
            if env.actions[i,2] > 0:
                table_og_tmp = table_og_tmp.rot90(1,[0, 1])
            start_ind_x = max(env.actions_origin[i][0]-1,0)
            end_ind_x = min(env.actions_origin[i][0]+2,env.cfg.og_resolution.tabletop[0])
            start_ind_y = max(env.actions_origin[i][1]-5,0)
            end_ind_y = min(env.actions_origin[i][1],env.cfg.og_resolution.tabletop[1])
            
            if torch.sum(table_og_tmp[env.actions_origin[i][0],start_ind_y:end_ind_y])>=1:
            # if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>=1:
                start_ind_y = max(env.actions_origin[i][1]-1,0)
                end_ind_y = min(env.actions_origin[i][1]+1,env.cfg.og_resolution.tabletop[1])
                # print('sum')
                # print(torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y]))
                if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>0:
                    # print('overlap')
                    min_reward = 0.0
                    for j in range(start_ind_x,end_ind_x):
                        if torch.sum(table_og_tmp[j,start_ind_y:end_ind_y])>0:
                            # print('pushing reward')
                            reward_near[i] = -3
                            # print(reward_near)
                            ################# modified in Dec 10
                            ind_pre = torch.where(env.table_og_pre[i]<=0.5)
                            ind_pre_x = ind_pre[0].clone()
                            ind_pre_y = ind_pre[1].clone()
                            ind_tmp = torch.where(ind_pre_y>=int(start_ind_y))
                            ind_tmp = ind_tmp[0]
                            # print(torch.mean(env.table_og_pre[i][ind_pre_x,ind_pre_y]))
                            # print(ind_pre_y)
                            # print(ind_pre_y.size())
                            # print(ind_tmp)
                            # print(ind_tmp[0].size())
                            # print(end_ind_y)
                            # print(torch.min(ind_pre_y[ind_tmp[0]].clone()))
                            if len(ind_tmp)>0:
                                ind_pre_y = ind_pre_y[ind_tmp].clone()
                                # print(ind_pre_y)
                                ind_pre_x = ind_pre_x[ind_tmp].clone()
                                ind_tmp = torch.where(ind_pre_x==int(j))
                                # print(env.actions_origin[i][0])
                                # print(ind_tmp)
                                # print(ind_pre_x[ind_tmp[0]].clone())
                                ind_tmp = ind_tmp[0]
                                if len(ind_tmp)>0:
                                    ind_pre_y = ind_pre_y[ind_tmp].clone()
                                    # print(ind_pre_y)
                                    ind_pre_y = ind_pre_y.clone() - int(env.actions_origin[i][1])
                                    reward_near[i] -= np.min(np.abs(ind_pre_y.cpu().numpy()))*0.5
                                    # print(np.min(np.abs(ind_pre_y.cpu().numpy())))
                                else:
                                    reward_near[i] -= 10*0.5
                            else:
                                reward_near[i] -= 10*0.5
                            # print(min_reward)
                            min_reward = min(float(reward_near[i]),min_reward)
                            # print(min_reward)
                    reward_near[i] = min_reward
                    #############################
                # else:
                #     print('found')
            else:
                start_ind_y = max(env.actions_origin[i][1]-1,0)
                end_ind_y = min(env.actions_origin[i][1]+1,env.cfg.og_resolution.tabletop[1])
                if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>0:
                    # print('overlap')
                    for j in range(start_ind_x,end_ind_x):
                        if torch.sum(table_og_tmp[j,start_ind_y:end_ind_y])>0:
                            # print('pushing reward')
                            reward_near[i] = -3
                            # print(reward_near)
                            ################# modified in Dec 10
                            ind_pre = torch.where(env.table_og_pre[i]<=0.5)
                            ind_pre_x = ind_pre[0].clone()
                            ind_pre_y = ind_pre[1].clone()
                            ind_tmp = torch.where(ind_pre_y>=int(start_ind_y))
                            ind_tmp = ind_tmp[0]
                            # print(torch.mean(env.table_og_pre[i][ind_pre_x,ind_pre_y]))
                            # print(ind_pre_y)
                            # print(ind_pre_y.size())
                            # print(ind_tmp)
                            # print(ind_tmp[0].size())
                            # print(end_ind_y)
                            # print(torch.min(ind_pre_y[ind_tmp[0]].clone()))
                            if len(ind_tmp)>0:
                                ind_pre_y = ind_pre_y[ind_tmp].clone()
                                # print(ind_pre_y)
                                ind_pre_x = ind_pre_x[ind_tmp].clone()
                                ind_tmp = torch.where(ind_pre_x==int(j))
                                # print(env.actions_origin[i][0])
                                # print(ind_tmp)
                                # print(ind_pre_x[ind_tmp[0]].clone())
                                ind_tmp = ind_tmp[0]
                                if len(ind_tmp)>0:
                                    ind_pre_y = ind_pre_y[ind_tmp].clone()
                                    # print(ind_pre_y)
                                    ind_pre_y = ind_pre_y.clone() - int(env.actions_origin[i][1])
                                    reward_near[i] -= np.min(np.abs(ind_pre_y.cpu().numpy()))*0.5
                                    # print(np.min(np.abs(ind_pre_y.cpu().numpy())))
                                else:
                                    reward_near[i] -= 10*0.5
                            else:
                                reward_near[i] -= 10*0.5
                            break
                    
                else:
                    reward_near[i] = -1
                # print('pushing reward')
                # reward_near[i] = -1
                # print(reward_near)
                    # if env.actions_origin[i][0]<=env.cfg.og_resolution.tabletop[0]-2 and env.actions_origin[i][0]>=1:
                    #     if env.actions_origin[i][1]<=env.cfg.og_resolution.tabletop[1]-2:
                            # table_og_tmp = env.table_og[i].clone()
                            # print(table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2])
                            # if torch.sum(table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2])==0:
                            #     reward_near[i] += 0.5
                            # table_og_tmp[env.actions_origin[i][0]-1:env.actions_origin[i][0]+2,env.actions_origin[i][1]-1:env.actions_origin[i][1]+2]=2
                            # table_og_tmp[env.actions_origin[i][0],env.actions_origin[i][1]-3:env.actions_origin[i][1]] = 2
                            # print('pushing position')
                            # plt.imshow(table_og_tmp.cpu().numpy())
                            # plt.show()
            show_nearest_point = np.zeros(2).astype(np.uint8)
            end_ind_y = min(env.actions_origin[i][1]+1,env.cfg.og_resolution.tabletop[1])
            start_ind_y = max(env.actions_origin[i][1]-5,0)
            if torch.sum(table_og_tmp[env.actions_origin[i][0],start_ind_y:end_ind_y]) == 0:
                ind_pre = torch.where(env.table_og_pre[i]>=0.8)
                ind_pre_x = ind_pre[0].clone()
                ind_pre_y = ind_pre[1].clone()
                # ind_tmp = torch.where(ind_pre_x==int(env.actions_origin[i][0]))
                # print(env.actions_origin[i][0])
                # print(ind_tmp)
                # print(ind_pre_x[ind_tmp[0]].clone())
                # ind_tmp = ind_tmp[0]
                # if len(ind_tmp)>0:
                #     ind_pre_x = ind_pre_x[ind_tmp].clone()
                #     ind_pre_y = ind_pre_y[ind_tmp].clone()
                    # print(ind_pre_y)
                    # print(ind_pre_x)
                '''method 2'''
                
                diff_tmp = torch.diff(table_og_tmp.clone())
                # plt.imshow(diff_tmp.cpu().numpy())
                # plt.show()
                ind_diff = torch.where(diff_tmp==-1)
                ind_diff_x = ind_diff[0].clone() 
                ind_diff_y = ind_diff[1].clone() +4
                for j in range(len(ind_diff_x)):
                    if ind_diff_y[j] >=49 and int(env.actions_origin[i][1])>25:
                        ind_diff_y[j] = 0
                    elif ind_diff_y[j] >=49 and int(env.actions_origin[i][1])<=25:
                        ind_diff_y[j] = 49
                ind_diff_x = ind_diff_x - int(env.actions_origin[i][0])
                ind_diff_y = ind_diff_y - int(env.actions_origin[i][1])
                # tmp_fig[ind_diff_x,ind_diff_y] = 1
                ind_diff_x = ind_diff_x.cpu().numpy()
                ind_diff_y = ind_diff_y.cpu().numpy()
                norm_dist = np.zeros(len(ind_diff_x))
                for j in range(len(ind_diff_x)):
                    norm_dist[j] = np.linalg.norm(np.array([ind_diff_x[j],ind_diff_y[j]]))
                ############################
                '''
                ind_tmp = torch.where(ind_pre_y<int(env.actions_origin[i][1]))
                ind_tmp = ind_tmp[0]
                if len(ind_tmp)>0:
                    # print(torch.max(ind_pre_y[ind_tmp].clone()))
                    ind_pre_x = ind_pre_x[ind_tmp].clone()-int(env.actions_origin[i][0])
                    ind_pre_y = ind_pre_y[ind_tmp].clone()-int(env.actions_origin[i][1])
                    ind_pre_x = ind_pre_x.cpu().numpy()
                    ind_pre_y = ind_pre_y.cpu().numpy()
                    norm_dist = np.zeros(len(ind_pre_x))
                    for j in range(len(ind_pre_x)):
                        norm_dist[j] = np.linalg.norm(np.array([ind_pre_x[j],ind_pre_y[j]]))

                else:
                    ind_pre_x = ind_pre_x-int(env.actions_origin[i][0]) # modified in Dec 10
                    ind_pre_y = ind_pre_y-int(env.actions_origin[i][1]) # modified in Dec 10
                    ################# modified in Dec 10
                    # ind_pre_x = np.ones(len(ind_pre_x))*25
                    # ind_pre_y = np.ones(len(ind_pre_x))*25
                    norm_dist = np.zeros(len(ind_pre_x))
                    for j in range(len(ind_pre_x)):
                        norm_dist[j] = np.linalg.norm(np.array([ind_pre_x[j],ind_pre_y[j]]))
                '''
                # else:
                #     # ind_pre_x = ind_pre_x-int(env.actions_origin[i][0]) # modified in Dec 10
                #     # ind_pre_y = ind_pre_y-int(env.actions_origin[i][1]) # modified in Dec 10
                #     ################# modified in Dec 10
                #     ind_pre_x = np.ones(len(ind_pre_x))*25
                #     ind_pre_y = np.ones(len(ind_pre_x))*25
                    #################################
                # print(np.sqrt(np.min(np.abs(ind_pre_y))**2+np.min(np.abs(ind_pre_x))**2))
                reward_near[i] -= np.min(np.abs(norm_dist))*0.5
                # print(np.min(np.abs(norm_dist)),np.argmin(np.abs(norm_dist)),ind_diff_x[int(np.argmin(np.abs(norm_dist)))],
                #       ind_diff_y[int(np.argmin(np.abs(norm_dist)))])
                # show_nearest_point= np.array([ind_diff_x[int(np.argmin(np.abs(norm_dist)))]+int(env.actions_origin[i][0]),
                #                               ind_diff_y[int(np.argmin(np.abs(norm_dist)))]+int(env.actions_origin[i][1])])
            
            ## visualize
            # tmp_fig = torch.zeros((env.cfg.og_resolution.tabletop[0],env.cfg.og_resolution.tabletop[1]),device=env.device)
            
            # diff_tmp = torch.diff(table_og_tmp.clone())
            # ind_diff = torch.where(diff_tmp==-1)
            # ind_diff_x = ind_diff[0].clone()
            # ind_diff_y = ind_diff[1].clone()+4
            # for j in range(len(ind_diff_x)):
            #     if ind_diff_y[j] >=49 and int(env.actions_origin[i][1])>25:
            #         ind_diff_y[j] = 0
            #     elif ind_diff_y[j] >=49 and int(env.actions_origin[i][1])<=25:
            #         ind_diff_y[j] = 49
            # tmp_fig2 = tmp_fig.clone()
            # tmp_fig2[:,:-1] = diff_tmp.clone()
            # tmp_fig[ind_diff_x,ind_diff_y] = 1
            # fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(10, 5))
            # start_ind_y = max(env.actions_origin[i][1]-1,0)
            # table_og_tmp_t = table_og_tmp.clone()
            # ax1.imshow(table_og_tmp_t.cpu().numpy())
            # table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 3
            # table_og_tmp[int(show_nearest_point[0]),int(show_nearest_point[1])] = 3
            # table_og_tmp[int(env.actions_origin[i][0]),int(env.actions_origin[i][1])] = 2
            # ax2.imshow(table_og_tmp.cpu().numpy())
            # tmp_fig2[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 1
            # tmp_fig2[int(env.actions_origin[i][0]),int(env.actions_origin[i][1])] = 2
            # ax3.imshow(tmp_fig2.cpu().numpy())
            # ax4.imshow(tmp_fig.cpu().numpy())
            # plt.show()
            # print('pushing position')
            # print(reward_near)
        return reward_near
    def reward_reaching(self,env:PushEnv):
        reward_near = torch.zeros((env.num_envs,),device=self.device)
        reward_near = torch.where(env.check_reaching>0.5,0,-1)
        
                        
        return reward_near


    def reward_for_toy_example(self,env:PushEnv):
        reward_toy = torch.zeros((env.num_envs,),device=self.device)
        actions_tmp = env.actions_origin.clone().cpu().numpy().astype(np.uint8)
        patches_tmp = env.random_patches.clone().cpu().numpy()
        actions_ori = env.actions_ori.clone().cpu().numpy()
        action_tmp_pre = env.previous_actions.clone().cpu().numpy()
        for i in range(env.num_envs):
            ind_x_pre = action_tmp_pre[i][0]
            ind_y_pre = action_tmp_pre[i][1]
            ind_x = actions_tmp[i][0]
            ind_y = actions_tmp[i][1]
            if patches_tmp[i,ind_x,ind_y] == 1:
                reward_toy[i] = 5
            else:
                reward_toy[i] = -1
            ''' toy version1
                ind_occ = np.where(patches_tmp[i]>=0.8)
                ind_occ_x = np.abs(ind_occ[0]-ind_x)
                ind_occ_y = np.abs(ind_occ[1]-ind_y)
                # nearest_x = np.min(ind_occ_x)
                # nearest_y = np.min(ind_occ_y)
                nearest_x = np.mean(ind_occ_x)
                nearest_y = np.mean(ind_occ_y)
                reward_toy[i] = -np.sqrt(nearest_x**2+nearest_y**2)/5
            if ind_x_pre + 4*(actions_ori[i][0]-1) < 0 or ind_x + 4*(actions_ori[i][0]-1) >= env.cfg.og_resolution.tabletop[0]:
                reward_toy[i] -=2
            if ind_y_pre + 4*(actions_ori[i][1]-1) < 0 or ind_y + 4*(actions_ori[i][1]-1) >= env.cfg.og_resolution.tabletop[1]:
                reward_toy[i] -=2
        reward_toy -= env.delta_same_action
            '''
        
        #     patches_tmp[i][ind_x,ind_y] = 1
        #     plt.imshow(patches_tmp[i])
        #     plt.show()
        # print(reward_toy)
        # print(env.delta_same_action)
        # print(actions_tmp)
        # print(action_tmp_pre)
        return reward_toy
                
    # def penalizing_tool_action_l2(self, env: PushEnv):
    #     """Penalize large values in action commands for the tool."""
    #     return -torch.square(env.actions[:, -1])

    # def tracking_object_position_exp(self, env: PushEnv, sigma: float, threshold: float):
    #     """Penalize tracking object position error using exp-kernel."""
    #     # distance of the end-effector to the object: (num_envs,)
    #     error = torch.sum(torch.square(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w), dim=1)
    #     # rewarded if the object is lifted above the threshold
    #     return (env.object.data.root_pos_w[:, 2] > threshold) * torch.exp(-error / sigma)

    # def tracking_object_position_tanh(self, env: PushEnv, sigma: float, threshold: float):
    #     """Penalize tracking object position error using tanh-kernel."""
    #     # distance of the end-effector to the object: (num_envs,)
    #     distance = torch.norm(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
    #     # rewarded if the object is lifted above the threshold
    #     return (env.object.data.root_pos_w[:, 2] > threshold) * (1 - torch.tanh(distance / sigma))

    # def lifting_object_success(self, env: PushEnv, threshold: float):
    #     """Sparse reward if object is lifted successfully."""
    #     return torch.where(env.object.data.root_pos_w[:, 2] > threshold, 1.0, 0.0)