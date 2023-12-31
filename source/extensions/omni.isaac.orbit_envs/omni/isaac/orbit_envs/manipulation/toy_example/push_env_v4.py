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
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
import pickle as pkl
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
from .push_cfg_v4 import PushEnvCfg, RandomizationCfg, YCBobjectsCfg, CameraCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.core.prims import RigidPrim,GeometryPrim
# from omni.isaac.orbit.utils.math import 
from omni.isaac.orbit.utils.array import convert_to_torch
import scipy.spatial.transform as tf
from .place_new_obj import place_new_obj_fun,get_new_obj_contour_bbox,draw_bbox
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
        
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        self.obj_handle_list = [f'handle_{i}' for i in range(int(1*len(ycb_name)))]
        # print(self.obj_handle_list)
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.obj1 = []
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
            self.obj1.append(RigidObject(obj_cfg1))
            self.obj_name_list.append(ycb_name[0]+str(i))
        
        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = PushObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = PushRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"]
        # print("num_obs")
        # print(num_obs)
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
        
        self.obj_on_table = dict()
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
        
        # camera
        position_camera = [0, 0, 1.5]
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
            self.env_i_tmp = 0
        self.reset_f = True
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)

        ''' modified for toy example
        # self.reset_objs(env_ids=env_ids)
        # self._randomize_table_scene(env_ids=env_ids)
        '''
        '''modified for toy example v2'''
        self.reset_objs(env_ids=env_ids)
        self._randomize_table_scene(env_ids=env_ids)
        '''modified for toy example v2'''
        self.new_obj_mask = np.zeros((self.cfg.og_resolution.tabletop[1],self.cfg.og_resolution.tabletop[0]))
        
        for _ in range(30):
            self.sim.step()
        self._update_table_og()
        self.table_og_pre[env_ids] = self.table_og[env_ids].clone()
        self.table_expand_og_pre[env_ids] = self.table_expand_og[env_ids].clone()
        self.falling_obj[env_ids] = 0
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
        self.falling_obj[env_ids] = 0
        self.stop_pushing[env_ids] = 0
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
        

    def _get_obj_mask(self,env_ids: VecEnvIndices):
        if np.sum(self.new_obj_mask) ==0:
            self.new_obj_mask = self.cfg.obj_mask.mask['sugarBox'] 
        mask = np.zeros((self.cfg.og_resolution.tabletop[1],self.cfg.og_resolution.tabletop[0]))
        s_x_ind = int(self.cfg.og_resolution.tabletop[1]/2-self.new_obj_mask.shape[1]/2)
        e_x_ind = int(self.cfg.og_resolution.tabletop[1]/2+self.new_obj_mask.shape[1]/2)
        s_y_ind = int(self.cfg.og_resolution.tabletop[0]/2-self.new_obj_mask.shape[0]/2)
        e_y_ind = int(self.cfg.og_resolution.tabletop[0]/2+self.new_obj_mask.shape[0]/2)
        mask[s_x_ind:e_x_ind,s_y_ind:e_y_ind] = self.new_obj_mask
        for j in env_ids.tolist():
            # print("mask j")
            # print(env_ids[j])
            self.obj_masks[j] = torch.from_numpy(mask).to(self.device)
        # plt.imshow(self.new_obj_mask)
        # plt.draw()
        # plt.imshow(mask)
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
        self.previous_actions = self.actions_origin.clone()
        self.table_og_pre = self.table_og.clone()
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
        self.actions_origin = actions.clone()
        self.action_ori = actions.clone()
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
        # print('action')
        # print(actions)
        # print(self.actions_origin)
        ''' modified for toy example'''
        self.actions = self.actions_origin.clone()
        self.actions = self.actions.type(torch.float16)
        actions_tmp = self.actions.clone()
        # self.actions[:,2] = self.actions[:,2]/8.0
        action_range = (float(self.cfg.og_resolution.tabletop[0])/200.0,float(self.cfg.og_resolution.tabletop[1])/200.0)
        self.actions[:,1] = action_range[1]*(actions_tmp[:,0].clone()-float(self.cfg.og_resolution.tabletop[1]/2))/float(self.cfg.og_resolution.tabletop[1]/2)
        self.actions[:,0] = action_range[0]*(-actions_tmp[:,1].clone()+float(self.cfg.og_resolution.tabletop[0]/2))/float(self.cfg.og_resolution.tabletop[0]/2)
        # print(self.actions)
        ################# stop pushing
        # for i in range(self.num_envs):
        #     if self.actions[i,1] >=0.235:
        #         if self.actions[i,0]>=0.235:
        #             if self.actions[i,2]>=0.240 and self.actions[i,2]<=0.36:
        #                 print("stop actions")
        #                 print(self.actions)
        #                 self.stop_pushing[i] = 1  
        #                 self.actions[i,1] = -0.5
        #                 self.actions[i,0] = 0.5

        actions_tmp = torch.zeros((self.num_envs,self._ik_controller.num_actions),device=self.device)
        actions_tmp[:,:2] = self.actions[:,:2].clone()
        # actions_tmp[:,:3] = self.actions[:,:3].clone()
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
        '''
        ########### lift the gripper above the start position
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
            vec_tmp[0] = 0.03*np.cos(2*np.pi*actions_tmp[i,2].cpu().numpy())
            vec_tmp[1] = 0.03*np.sin(2*np.pi*actions_tmp[i,2].cpu().numpy())
            
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
            vec_tmp[0] = 0.03*np.cos(2*np.pi*actions_tmp[i,2].cpu().numpy())
            vec_tmp[1] = 0.03*np.sin(2*np.pi*actions_tmp[i,2].cpu().numpy())
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
            vec_tmp[0] = 0.02*np.cos(2*np.pi*actions_tmp[i,2].cpu().numpy())
            vec_tmp[1] = 0.02*np.sin(2*np.pi*actions_tmp[i,2].cpu().numpy())
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
        '''
        # env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
        # dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        # self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # self.robot.update_buffers(self.dt)
        # for _ in range(70):
        #     self.sim.step()
        for i,obj_t in enumerate(self.obj1):
            obj_t.update_buffers(self.dt)
        ''' modified for toy example
        self._check_fallen_objs(env_ids)
        # check_placing
        self._check_placing()
        '''
        # reward
        self.reward_buf = self._reward_manager.compute()
        # print("reward")
        # print(self._reward_manager.compute())
        # print(self.reward_buf)
        # terminations
        ''' only for toy example'''
        self._check_reaching_toy_v2()
        ''' only for toy example'''
        self._check_termination()
        self.delta_same_action = torch.where(torch.sum(torch.abs(self.previous_actions-self.actions_origin),dim=1)<=0.1,1,0)

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

        ''' modified for toy example
        # self.extras["is_success"] = torch.where(self.place_success>=0.5, 1, self.reset_buf)
        # # print(self.extras["time_outs"])

        # for i,value_timeout in enumerate(self.extras['time_outs']):
        #     if value_timeout:
        #         if self.place_success[i]>=0.5:
        #             self.extras["time_outs"][i] = False
        '''
        ''' modified for toy example'''
        self.extras["is_success"] = torch.where(self.check_reaching>=0.5, 1, self.reset_buf)
        # print(self.extras["time_outs"])
        for i,value_timeout in enumerate(self.extras['time_outs']):
            if value_timeout:
                if self.check_reaching[i]>=0.5:
                    self.extras["time_outs"][i] = False
        ''' modified for toy example'''
        # -- update USD visualization
        self._update_table_og()
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
            self.num_actions = 2
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_w)
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
        # objects_pcd = o3d.geometry.PointCloud()
        # objects_pcd.points = o3d.utility.Vector3dVector(inliers)
        pts_tab = np.array(inliers)
        # o3d.visualization.draw_geometries([objects_pcd])
        # print(camera.data.output["distance_to_image_plane"].shape)
        # print(pointcloud_w.shape)
        select_m = np.dot(inliers,plane_model) + float(plane_model_ori[3])
        index_objects = np.argwhere((select_m>=0.005)).reshape(-1).astype(int)
        pts = inliers[index_objects].copy()
        Nx = self.cfg.og_resolution.tabletop[0]
        Ny = self.cfg.og_resolution.tabletop[1]
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
        occupancy = np.fliplr(occupancy)
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
        Nx = self.cfg.og_resolution.tabletop[0]+12
        Ny = self.cfg.og_resolution.tabletop[1]+12
        # objects_pcd = o3d.geometry.PointCloud()
        # objects_pcd.points = o3d.utility.Vector3dVector(whole_point_cloud)
        # o3d.visualization.draw_geometries([objects_pcd])
        ################ original table
        occupancy_ex = np.zeros( (Ny,Nx) )
        occupancy_ex[6:self.cfg.og_resolution.tabletop[1]+6,6:self.cfg.og_resolution.tabletop[0]+6] = 1
        u = (pts_ex[:,0] - np.min(pts_tab[:,0]))/ ( np.max(pts_tab[:,0])-np.min(pts_tab[:,0]) )
        v = (pts_ex[:,1] - np.min(pts_tab[:,1]))/ ( np.max(pts_tab[:,1])-np.min(pts_tab[:,1]) )
        u = (Nx-12-1)*u +6
        v = (Ny-12-1)*v +6
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
        occupancy_ex[v,u] = 2
        # plt.imshow(occupancy_ex)
        # plt.show()
        
        
        return occupancy,occupancy_ex

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
    def _check_fallen_objs(self,env_ids:VecEnvIndices):
        torch_fallen = torch.zeros((self.num_envs,),device=self.device)
        for k in env_ids.tolist():
            if k in self.obj_on_table:
                for _,obj in enumerate(self.obj_on_table[k]):
                # print(obj.data.root_pos_w[1, :3])
                # if obj != 0:
                    torch_fallen[k] += torch.where(obj.data.root_pos_w[k, 2] < -0.05, 1, 0)
        # print(torch_fallen)
        # print(self.falling_obj_all)
        self.falling_obj[env_ids] = torch_fallen[env_ids] - self.falling_obj_all[env_ids]
        self.falling_obj_all[env_ids] = torch_fallen[env_ids]
        
    def _update_table_og(self):
        if self.reset_f:
            for i in range(self.num_envs):
                self.cams[i].update(self.dt)
                
                og,og_ex = self.get_og(self.cams[i])
                
                self.table_og[i] = torch.from_numpy(og.copy()).to(self.device)
                self.table_expand_og[i] = torch.from_numpy(og_ex.copy()).to(self.device)
                

    def _check_placing(self):
        env_ids=torch.from_numpy(np.arange(self.num_envs)).to(self.device)
        self._update_table_og()
        # for i in range(self.num_envs):
        #     plt.imshow()
        for i in range(self.num_envs):
            occupancy = self.table_og[i].cpu().numpy()
            vertices_new_obj = self.new_obj_vertices[i]
            flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(occupancy,vertices_new_obj)  
            # plt.imshow(occu_tmp)
            # plt.show()
            if flag_found:
                self.place_success[i] = 1
                # plt.imshow(self.obj_masks[i].cpu().numpy())
                # plt.show()
                ############## visulaize placing
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
        self.check_reaching = torch.zeros((self.num_envs,),device=self.device)
        action_tmp_reward = self.actions.clone().cpu().numpy().astype(np.uint8)
        for i in range(self.num_envs):
            table_og_tmp = self.table_og_pre[i].clone()
            start_ind_x = max(self.actions_origin[i][0]-1,0)
            end_ind_x = min(self.actions_origin[i][0]+2,self.cfg.og_resolution.tabletop[0])
            start_ind_y = max(self.actions_origin[i][1]-4,0)
            end_ind_y = min(self.actions_origin[i][1],self.cfg.og_resolution.tabletop[1])
            if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>=1:
                start_ind_y = max(self.actions_origin[i][1]-1,0)
                end_ind_y = min(self.actions_origin[i][1]+1,self.cfg.og_resolution.tabletop[1])
                if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])==0:
                    self.check_reaching[i] = 1   

    def _check_termination(self) -> None:
        # access buffers from simulator
        # object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when stop pushing
        if self.cfg.terminations.stop_pushing:
            self.reset_buf = torch.where(self.stop_pushing >= 0.5, 1, self.reset_buf)
        # -- when task is successful
        if self.cfg.terminations.is_success:
            ''' modified because of toy example
            self.reset_buf = torch.where(self.place_success >= 0.5, 1, self.reset_buf)
            '''
            self.reset_buf = torch.where(self.check_reaching >= 0.5, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
    
    def reset_objs(self,env_ids: torch.Tensor):
        for i,obj_t in enumerate(self.obj1):
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
        # self.obj_on_table = []
        num_env = len(file_name)
        choosen_env_id = np.random.randint(0,num_env)
        # choosen_env_id = self.env_i_tmp
        # print(file_name[choosen_env_id],env_ids,self.env_i_tmp)
        env_path = "generated_table/"+file_name[choosen_env_id]
        # env_path = "generated_table/dict_20.pkl"
        if self.env_i_tmp <num_env-1:
            self.env_i_tmp +=1
        fileObject2 = open(env_path, 'rb')
        env =  pkl.load(fileObject2)
        obj_pos_rot = env[0]
        
        # self.new_obj_mask = self.cfg.obj_mask.mask["tomatoSoupCan"]
        self.new_obj_mask = self.cfg.obj_mask.mask[env[1]]
        
        for i in env_ids.tolist():
            self.new_obj_type[int(i)] = env[1]
        # self.new_obj_type = "tomatoSoupCan"
        fileObject2.close()
        # print(env_ids)
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
    def penalizing_pushing_outside(self,env:PushEnv):
        pixel_outside_table = torch.zeros((env.num_envs,),device=self.device)
        env_tab_ex_tmp = env.table_expand_og.clone()
        env_tab_ex_tmp_pre = env.table_expand_og_pre.clone()
        for i in range(env.num_envs):
            env_tab_ex_tmp[i][6:env.cfg.og_resolution.tabletop[1]+6,
                           6:env.cfg.og_resolution.tabletop[0]+6] = 0
            env_tab_ex_tmp_pre[i][6:env.cfg.og_resolution.tabletop[1]+6,
                           6:env.cfg.og_resolution.tabletop[0]+6] = 0
            pixel_outside_table[i] = torch.sum(env_tab_ex_tmp[i]-env_tab_ex_tmp_pre[i])
        # print(-pixel_outside_table.type(torch.float16)/float(300.0))
        return -pixel_outside_table.type(torch.float16)/float(300.0)
    def penalizing_repeat_actions(self,env:PushEnv):
        # print("repeat")
        # print(env.delta_same_action)
        return -env.delta_same_action.type(torch.float16) 
    def penalizing_falling(self,env:PushEnv):
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
        return env.place_success.type(torch.float16)
    def penalizing_steps(self,env:PushEnv):
        # print("pernalize steps")
        # print(-torch.where(env.step_count!=0,1,0).to(env.device))
        # print('steps')
        # print(torch.where(env.step_count!=0,1,0).to(env.device))
        return -torch.where(env.step_count!=0,1,0).to(env.device).type(torch.float16)
    def reward_og_change(self,env:PushEnv):
        delta_og = torch.zeros((env.num_envs,),device=self.device)
        for i in range(env.num_envs):
            delta_tmp = env.table_og[i].clone() - env.table_og_pre[i].clone()
            if torch.sum(torch.abs(delta_tmp))>30:
                if torch.sum(torch.abs(delta_tmp))<60:
                    delta_og[i] = 1
                else:
                    delta_og[i] = 2
        # env.table_og_pre = env.table_og.clone()
        # print("reward og")
        # print(delta_og)
        # print('og change')
        # print(delta_og)
        return delta_og.type(torch.float16)
    def reward_near_obj_1(self,env:PushEnv):
        print('reward 1')
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
            start_ind_x = max(env.actions_origin[i][0]-1,0)
            end_ind_x = min(env.actions_origin[i][0]+2,env.cfg.og_resolution.tabletop[0])
            start_ind_y = max(env.actions_origin[i][1]-5,0)
            end_ind_y = min(env.actions_origin[i][1],env.cfg.og_resolution.tabletop[1])
            
            if torch.sum(table_og_tmp[env.actions_origin[i][0],start_ind_y:end_ind_y])>=1:
            # if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>=1:
                start_ind_y = max(env.actions_origin[i][1]-1,0)
                end_ind_y = min(env.actions_origin[i][1]+1,env.cfg.og_resolution.tabletop[1])
                if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y])>0:
                    # print('pushing reward')
                    reward_near[i] = -2
                    # print(reward_near)
            else:
                # print('pushing reward')
                reward_near[i] = -2
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
            
            end_ind_y = min(env.actions_origin[i][1]+1,env.cfg.og_resolution.tabletop[1])
            start_ind_y = max(env.actions_origin[i][1]-5,0)
            if torch.sum(table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y]) == 0:
                ind_pre = torch.where(env.table_og_pre[i]>=0.8)
                ind_pre_x = ind_pre[0].clone().cpu().numpy().reshape(-1)
                ind_pre_y = ind_pre[1].clone().cpu().numpy().reshape(-1)
                ind_tmp = np.array(np.where(ind_pre_y<=int(end_ind_x))).reshape(-1).astype(np.uint8)
                if len(ind_tmp)>0:
                    ind_pre_x = ind_pre_x[ind_tmp].copy()-int(env.actions_origin[i][0])
                    ind_pre_y = ind_pre_y[ind_tmp].copy()-int(env.actions_origin[i][1])
                else:
                    ind_pre_x = ind_pre_x-int(env.actions_origin[i][0])
                    ind_pre_y = ind_pre_y-int(env.actions_origin[i][1])
                # print(np.sqrt(np.min(np.abs(ind_pre_y))**2+np.min(np.abs(ind_pre_x))**2))
                reward_near[i] -= 0.1*np.sqrt(np.min(np.abs(ind_pre_y))**2+np.min(np.abs(ind_pre_x))**2)
            
            ### visualize
            # start_ind_y = max(env.actions_origin[i][1]-1,0)
            # table_og_tmp[start_ind_x:end_ind_x,start_ind_y:end_ind_y] = 3
            # print('pushing position')
            # print(reward_near)
            # plt.imshow(table_og_tmp.cpu().numpy())
            # plt.show()
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