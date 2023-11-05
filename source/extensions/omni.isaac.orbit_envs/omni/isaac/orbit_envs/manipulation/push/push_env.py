# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import gym.spaces
import math
import torch
from typing import List
import numpy as np
import open3d as o3d
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
import pickle as pkl
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.core.objects import FixedCuboid
from .push_cfg import PushEnvCfg, RandomizationCfg, YCBobjectsCfg, CameraCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.core.prims import RigidPrim,GeometryPrim
from omni.isaac.orbit.utils.math import convert_quat
import scipy.spatial.transform as tf
class PushEnv(IsaacEnv):
    """Environment for lifting an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: PushEnvCfg = None, **kwargs):
        
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.object = RigidObject(cfg=self.cfg.object)

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
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        # print("num_obs")
        # print(num_obs)
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # flag to check whether the new object is placed or not
        self.success_place = False
        # Take an initial step to initialize the scene.
        # This is required to compute quantities like Jacobians used in step().
        self.sim.step()
        # -- fill up buffers
        self.object.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)
        self.obj_list = []
    """
    Implementation specifics.
    """
    
    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-.5)
        # table
        # prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.usd_path)
        # prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.table_path,position=(0,0,-0.25),scale=(1,0.6,0.5))
        # Table = GeometryPrim(self.template_env_ns + "/Table",collision=True)
        # Table = RigidPrim(self.template_env_ns + "/Table",mass=100000)
        self.Table = FixedCuboid(self.template_env_ns + "/Table",position=(0,0,-0.25),scale=(1,0.6,0.5))
        self.Table.set_collision_enabled(True)
        self.Table.set_collision_approximation("convexHull")
        # prim_utils.create_prim(self.template_env_ns + "/sideTable", usd_path=self.cfg.table.table_path,position=(0.35,-0.9,-0.3),scale=(0.4,0.4,0.4))
        # sideTable = GeometryPrim(self.template_env_ns + "/sideTable",collision=True)
        # sideTable = RigidPrim(self.template_env_ns + "/sideTable",mass=10000)
        self.sideTable = FixedCuboid(self.template_env_ns + "/sideTable",position=(0.35,-0.9,-0.3),scale=(0.4,0.4,0.4))
        self.sideTable.set_collision_enabled(True)
        self.sideTable.set_collision_approximation("convexHull")
        prim_utils.create_prim(self.template_env_ns + "/Robotbase", usd_path=self.cfg.table.table_path,position=(0,-0.45,-0.25),scale=(0.3,0.26,0.5))
        # GeometryPrim(self.template_env_ns + "/Robotbase",collision=True)
        # RigidPrim(self.template_env_ns + "/Robotbase",mass=10000)
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot",translation=(0.0, -.45, 0))
        # object
        self.object.spawn(self.template_env_ns + "/Object")
        # camera
        position_camera = [0, 0, 1.7]
        orientation = [1, 0, 0, 0]
        position_handcamera = [0.35,-0.9,0.8]
        self.camera = Camera(cfg=self.cfg.camera.camera_cfg, device='cuda')
        # self.hand_camera = Camera(cfg=self.cfg.camera.camera_cfg,device='cuda')
        # hand_camera.spawn("/World/Robot/panda_hand/hand_camera", translation=(0.1, 0.0, 0.0),orientation=(0,0,1,0))
        # self.hand_camera.spawn(self.template_env_ns + "/hand_camera",translation=position_handcamera,orientation=orientation)
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
        # self._randomize_table_scene()  
        self._set_table_scene()      
        # return list of global prims
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- object pose
        self.env_obj = dict()
        self._randomize_object_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_initial_pose)
        # -- goal pose
        self._randomize_object_desired_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_desired_pose)
        

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # print("env_ids")
        # print(env_ids)
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            self._ik_controller.set_command(self.actions[:, :-1])
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
            self.robot_actions[:, -1] = self.actions[:, -1]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:] = self.actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.object.update_buffers(self.dt)
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- add information to extra if task completed
        object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
        self.extras["is_success"] = torch.where(object_position_error < 0.02, 1, self.reset_buf)
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
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
        self.sim.reset()
        
        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.object.initialize(self.env_ns + "/.*/Object")
        print("camera")
        print(self.camera)
        # self.camera.initialize()
        # self.hand_camera.initialize()
        # self.camera.update(self.dt)
        # self.hand_camera.update(self.dt)
        self.cams = [self.camera] + [Camera(cfg=self.cfg.camera.camera_cfg,device='cpu') for _ in range(self.num_envs - 1)]
        # self.hand_cams = [self.hand_camera] + [Camera(cfg=self.cfg.camera.camera_cfg, device = 'cpu') for _ in range(self.num_envs - 1)]
        # self.get_pcd(self.camera)
        for i in range(self.num_envs):
           self.cams[i].initialize(self.env_ns + f"/env_{i}/CameraSensor/Camera")
        #    env_pos = np.array(self.envs_positions[i].cpu().numpy())
        #    self.cams[i].set_world_pose_from_view(eye=np.array([0, 0, 1.8]) + env_pos, target=np.array([0, 0, 0]) + env_pos)
        #    self.cams[i].update(self.dt)
        #    self.hand_cams[i].initialize(self.env_ns + f"/env_{i}/hand_camera/Camera")
        # #    env_pos = np.array(self.envs_positions[i].cpu().numpy())
        #    self.hand_cams[i].set_world_pose_from_view(eye=np.array([0.35,-0.9,0.8]) + env_pos, target=np.array([0.35, -0.9, 0]) + env_pos)
        #    self.hand_cams[i].update(self.dt)
        
        # self.camera.initialize(self.env_ns + "/.*/CameraSensor/Camera")
        # self.hand_camera.initialize(self.env_ns + "/.*/hand_camera")
        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.object_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        # buffers
        self.object_root_pose_ee = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
    def get_pcd(self,camera):
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
        return pcd
    
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

    def _check_termination(self) -> None:
        # access buffers from simulator
        object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when task is successful
        if self.cfg.terminations.is_success:
            object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
            self.reset_buf = torch.where(object_position_error < 0.02, 1, self.reset_buf)
        # -- object fell off the table (table at height: 0.0 m)
        if self.cfg.terminations.object_falling:
            self.reset_buf = torch.where(object_pos[:, 2] < -0.05, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
    # def take_image(self,env_ids:torch.Tensor):


    def _randomize_table_scene(self):
        
        
        ycb_usd_paths = self.cfg.YCBdata.ycb_usd_paths
        ycb_name = self.cfg.YCBdata.ycb_name
        self.obj_dict = dict()
        num_obj = np.random.randint(1,5)
        if num_obj >=1:
            for _ in range(num_obj):
                randi = np.random.randint(0,len(ycb_name))
                angle = np.random.randint(0,180)
                # angle = 0
                key_ori = ycb_name[randi]
                # key_ori = "mug"
                usd_path = ycb_usd_paths[key_ori]
                if key_ori not in self.obj_dict:
                    self.obj_dict[key_ori] = 1
                else:
                    self.obj_dict[key_ori] +=1
                key = key_ori+str(self.obj_dict[key_ori])
                translation = torch.rand(3).tolist()
                translation = [translation[0]*0.8-0.4,0.45*translation[1]-0.225,0.1]
                # translation = [0,0,0.2]
                print(translation,angle,key_ori)
                rot = convert_quat(tf.Rotation.from_euler("XYZ", (0,0,angle), degrees=True).as_quat(), to="wxyz")
                if key_ori in ["mug","tomatoSoupCan","pitcherBase","tunaFishCan","bowl","banana"]:
                    rot = convert_quat(tf.Rotation.from_euler("XYZ", (-90,angle,0), degrees=True).as_quat(), to="wxyz")
                prim_utils.create_prim(self.template_env_ns+f"/{key}",usd_path=usd_path, translation=translation,orientation=rot)
                GeometryPrim(self.template_env_ns+f"/{key}",collision=True).set_collision_approximation("convexHull")
                RigidPrim(self.template_env_ns+f"/{key}",mass=0.3)
                # prim_utils.create_prim(f"/World/Objects/{key}", usd_path=usd_path, translation=translation,orientation=rot)
                # GeometryPrim(f"/World/Objects/{key}",collision=True)
                # RigidPrim(f"/World/Objects/{key}",mass=0.3)
                for _ in range(50):
                    self.sim.step()
    
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

    def _set_table_scene(self):
        file_name = self.cfg.env_name
        ycb_usd_paths = self.cfg.YCBdata.ycb_usd_paths
        ycb_name = self.cfg.YCBdata.ycb_name
        self.obj_dict = dict()
        self.obj_on_table = []
        num_env = len(file_name)
        choosen_env_id = np.random.randint(0,num_env)
        env_path = "generated_table/"+file_name[choosen_env_id]
        fileObject2 = open(env_path, 'rb')
        env =  pkl.load(fileObject2)
        obj_pos_rot = env[0]
        self.new_obj_mask = self.cfg.obj_mask.mask[env[1]]
        fileObject2.close()
        # print(env)
        for _ in obj_pos_rot:
            for k in obj_pos_rot[_]:
                # print(_)
                # print(ycb_usd_paths["largeClamp"])
                usd_path = ycb_usd_paths[_]
                # print(usd_path)
                if _ not in self.obj_dict:
                    self.obj_dict[_] = 1
                else:
                    self.obj_dict[_] +=1
                # print(self.obj_dict)
                key = _+str(self.obj_dict[_])
                prim_utils.create_prim(self.template_env_ns+f"/table_obj/{key}",usd_path=usd_path, translation=k[0],orientation=k[1])
                GeometryPrim(self.template_env_ns+f"/table_obj/{key}",collision=True).set_collision_approximation("convexHull")
                RigidPrim(self.template_env_ns+f"/table_obj/{key}",mass=0.3)
                self.obj_on_table.append(key)
                for j in range(5):
                    self.sim.step()

        # print(env)
        # print(new_obj_mask)
        return 0
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
        return env.tabletop_og
    def new_obj_mask(self,env:PushEnv):
        return env.new_obj_mask
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

    def object_positions(self, env: PushEnv):
        """Current object position."""
        return env.object.data.root_pos_w - env.envs_positions

    def object_orientations(self, env: PushEnv):
        """Current object orientation."""
        # make the first element positive
        quat_w = env.object.data.root_quat_w
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def object_relative_tool_positions(self, env:PushEnv):
        """Current object position w.r.t. end-effector frame."""
        return env.object.data.root_pos_w - env.robot.data.ee_state_w[:, :3]

    def object_relative_tool_orientations(self, env: PushEnv):
        """Current object orientation w.r.t. end-effector frame."""
        # compute the relative orientation
        quat_ee = quat_mul(quat_inv(env.robot.data.ee_state_w[:, 3:7]), env.object.data.root_quat_w)
        # make the first element positive
        quat_ee[quat_ee[:, 0] < 0] *= -1
        return quat_ee

    def object_desired_positions(self, env: PushEnv):
        """Desired object position."""
        return env.object_des_pose_w[:, 0:3] - env.envs_positions

    def object_desired_orientations(self, env: PushEnv):
        """Desired object orientation."""
        # make the first element positive
        quat_w = env.object_des_pose_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

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

    def reaching_object_position_l2(self, env: PushEnv):
        """Penalize end-effector tracking position error using L2-kernel."""
        return torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w), dim=1)

    def reaching_object_position_exp(self, env: PushEnv, sigma: float):
        """Penalize end-effector tracking position error using exp-kernel."""
        error = torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w), dim=1)
        return torch.exp(-error / sigma)

    def reaching_object_position_tanh(self, env: PushEnv, sigma: float):
        """Penalize tool sites tracking position error using tanh-kernel."""
        # distance of end-effector to the object: (num_envs,)
        ee_distance = torch.norm(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
        # distance of the tool sites to the object: (num_envs, num_tool_sites)
        object_root_pos = env.object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        tool_sites_distance = torch.norm(env.robot.data.tool_sites_state_w[:, :, :3] - object_root_pos, dim=-1)
        # average distance of the tool sites to the object: (num_envs,)
        # note: we add the ee distance to the average to make sure that the ee is always closer to the object
        num_tool_sites = tool_sites_distance.shape[1]
        average_distance = (ee_distance + torch.sum(tool_sites_distance, dim=1)) / (num_tool_sites + 1)

        return 1 - torch.tanh(average_distance / sigma)

    def penalizing_arm_dof_velocity_l2(self, env: PushEnv):
        """Penalize large movements of the robot arm."""
        return -torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_tool_dof_velocity_l2(self, env: PushEnv):
        """Penalize large movements of the robot tool."""
        return -torch.sum(torch.square(env.robot.data.tool_dof_vel), dim=1)

    def penalizing_arm_action_rate_l2(self, env: PushEnv):
        """Penalize large variations in action commands besides tool."""
        return -torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)

    def penalizing_tool_action_l2(self, env: PushEnv):
        """Penalize large values in action commands for the tool."""
        return -torch.square(env.actions[:, -1])

    def tracking_object_position_exp(self, env: PushEnv, sigma: float, threshold: float):
        """Penalize tracking object position error using exp-kernel."""
        # distance of the end-effector to the object: (num_envs,)
        error = torch.sum(torch.square(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w), dim=1)
        # rewarded if the object is lifted above the threshold
        return (env.object.data.root_pos_w[:, 2] > threshold) * torch.exp(-error / sigma)

    def tracking_object_position_tanh(self, env: PushEnv, sigma: float, threshold: float):
        """Penalize tracking object position error using tanh-kernel."""
        # distance of the end-effector to the object: (num_envs,)
        distance = torch.norm(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
        # rewarded if the object is lifted above the threshold
        return (env.object.data.root_pos_w[:, 2] > threshold) * (1 - torch.tanh(distance / sigma))

    def lifting_object_success(self, env: PushEnv, threshold: float):
        """Sparse reward if object is lifted successfully."""
        return torch.where(env.object.data.root_pos_w[:, 2] > threshold, 1.0, 0.0)
