# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")
enable_extension("omni.kit.window.viewport")
import os
import pickle as pkl
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.dynamic_control import _dynamic_control

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg
from omni.isaac.orbit.sensors.camera import PinholeCameraCfg
##
# Scene settings
##
@configclass
class CameraCfg:
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=300,
        width=300,
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

@configclass
class TableCfg:
    """Properties for the table."""
    table_path = f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd"
    # note: we use instanceable asset since it consumes less memory
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"


@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        scale=(0.8, 0.8, 0.8),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.9, 0.0, -0.1), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=1.5, dynamic_friction=1.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )

@configclass
class env_name:
    file_list = os.listdir("generated_table/")
@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [0.4, -0.25, 0.075]  # position (x,y,z)
        position_uniform_max = [0.6, 0.25, 0.075]  # position (x,y,z)

    @configclass
    class ObjectDesiredPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.0, 0.5]  # position default (x,y,z)
        position_uniform_min = [0.4, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.6, 0.25, 0.5]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        # -- joint state
        # table_scene = {"scale": 1.0} ######## original
        ''' modified for toy example v2'''
        table_scene = {"scale": 1.0}
        '''modified for toy example v2'''
        ''' modified for toy example toy example v1'''
        # obs_for_toy_example = {"scale": 1.0}
        '''modified for toy example toy example v1'''
        # new_obj_mask = {"scale": 1.0}
        # arm_dof_pos = {"scale": 1.0}
        # # arm_dof_pos_scaled = {"scale": 1.0}
        # # arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        # tool_dof_pos_scaled = {"scale": 1.0}
        # # -- end effector state
        # tool_positions = {"scale": 1.0}
        # tool_orientations = {"scale": 1.0}
        # # -- object state
        # # object_positions = {"scale": 1.0}
        # # object_orientations = {"scale": 1.0}
        # object_relative_tool_positions = {"scale": 1.0}
        # # object_relative_tool_orientations = {"scale": 1.0}
        # # -- object desired state
        # object_desired_positions = {"scale": 1.0}
        # # -- previous action
        # arm_actions = {"scale": 1.0}
        # tool_actions = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class YCBobjectsCfg:
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
    #     "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    #     "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    # }
    # ycb_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle']
    # ycb_usd_paths = {
    #     "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    #     "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    #     # "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    #     "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    # }
    # ycb_name = ['crackerBox','sugarBox','mustardBottle']
    ycb_usd_paths = {
        # "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        # "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        # "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }
    ycb_name = ['sugarBox']

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- robot-centric
    # reaching_object_position_l2 = {"weight": 0.0}
    # reaching_object_position_exp = {"weight": 2.5, "sigma": 0.25}
    # reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.1}
    # penalizing_arm_dof_velocity_l2 = {"weight": 1e-5}
    # penalizing_tool_dof_velocity_l2 = {"weight": 1e-5}
    # penalizing_robot_dof_acceleration_l2 = {"weight": 1e-7}
    # -- action-centric
    # penalizing_arm_action_rate_l2 = {"weight": 1e-2}
    ''' modified for toy example
    reward_og_change = {"weight":0.25}
    reward_distribution_closer = {"weight":0.25}
    check_placing = {"weight": 2}
    penalizing_falling = {"weight": 1}
    reward_near_obj = {"weight":0.1}
    '''
    '''modified for toy example'''
    '''reward for toy example v1'''
    #reward_for_toy_example = {"weight": 1}
    '''reward for toy example v1'''
    '''reward for toy example v2'''
    #reward_for_toy_example = {"weight": 1}
    '''reward for toy example v2'''
    # check_placing = {"weight": 2}
    # reward_near_obj = {"weight": 1}
    reward_reaching = {"weight": 1}
    # penaltizing_falling = {"weight": 1} ## DEc20_00-56-30
    # penaltizing_falling = {"weight": 2} ## Dec24_16-21-23 before
    # penaltizing_falling = {"weight": 3} ## Dec24_16-21-23 before
    # penaltizing_steps = {"weight": 0.1}
    # penaltizing_repeat_actions = {"weight": 0.5}
    # penaltizing_pushing_outside = {"weight":0.2} ## Dec22_20-44-43 before
    # penaltizing_pushing_outside = {"weight":0.5} ## Dec24_16-21-23 before
    # penaltizing_pushing_outside = {"weight":0.7}
    # reward_max_tsdf_increase = {"weight": 10}
    # penaltizing_stop = {"weight": 2} ## Dec26 after
    # penalizing_tool_action_l2 = {"weight": 1e-2}
    # -- object-centric
    # tracking_object_position_exp = {"weight": 5.0, "sigma": 0.25, "threshold": 0.08}
    # tracking_object_position_tanh = {"weight": 5.0, "sigma": 0.2, "threshold": 0.08}
    # lifting_object_success = {"weight": 3.5, "threshold": 0.08}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # stop_pushing = False ## Dec24_16-21-23 before
    stop_pushing = False 
    episode_timeout = True  # reset when episode length ended
    # object_falling = True  # reset when object falls off the table
    is_success = False # reset when object is placed modified for toy example

@configclass
class occupancy_grid_resolution:
    """resolution of the occupancy grid"""
    tabletop = [50,50]
    new_obj = [40,40]
@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    # control_type = "default"  # "default", "inverse_kinematics"
    control_type = "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 3

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        # command_type="pose_rel",
        # command_type = "pose_rel",
        command_type = "pose_abs",
        ik_method="dls",
        # position_command_scale=(0.1, 0.1, 0.1),
        # rotation_command_scale=(0.1, 0.1, 0.1),
        position_command_scale=(0.03, 0.03, 0.03),
        rotation_command_scale=(0.03, 0.03, 0.03),
    )


##
# Environment configuration
##
@configclass
class ObjMask:
    def __init__(self) -> None:
        self.mask = dict()
        file_list = os.listdir("obj_mask/")
        obj_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
                'banana','bowl','largeClamp','scissors']
        for i in range(len(file_list)):
            for j in range(len(obj_name)):
                if obj_name[j] in file_list[i]:
                    # print(obj_name[j],file_list[i])
                    fileObject2 = open('obj_mask/'+file_list[i], 'rb')
                    self.mask[obj_name[j]]=  pkl.load(fileObject2)

                    fileObject2.close()
                    



@configclass
class PushEnvCfg(IsaacEnvCfg):
    """Configuration for the push environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=3, episode_length_s=0.24)
    viewer: ViewerCfg = ViewerCfg(debug_vis=False, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=0.01,
        substeps=1,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 2, # 1024 * 1024 * 2
            gpu_total_aggregate_pairs_capacity= 1024 * 1024 * 2 *1,#16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,

            # gpu_max_rigid_contact_count=1024**2*2, #1024**2*2,
            # gpu_max_rigid_patch_count=160*2048*10, #160*2048*10, #160*2048*10,
            # gpu_found_lost_pairs_capacity = 1024 * 1024 * 2 * 1,#1024 * 1024 * 2 * 1, #1024 * 1024 * 2 * 8,
            # gpu_found_lost_aggregate_pairs_capacity=100,#1024 * 1024 * 32 * 1, #1024 * 1024 * 32,
            # gpu_total_aggregate_pairs_capacity=100, #1024 * 1024 * 2 *1, #1024 * 1024 * 2 * 8
            # friction_correlation_distance=0.0025,
            # friction_offset_threshold=0.04,
            # bounce_threshold_velocity=0.5,
            # gpu_max_num_partitions=8,
        ),
    )

    # Scene Settings
    # -- robot
    robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # -- table
    table: TableCfg = TableCfg()
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
    # Camera settings
    camera: CameraCfg = CameraCfg()
    YCBdata: YCBobjectsCfg = YCBobjectsCfg()
    # resolution of the occupancy grid
    og_resolution: occupancy_grid_resolution = occupancy_grid_resolution()
    obj_mask: ObjMask = ObjMask()
    env_name: env_name = env_name().file_list
    pre_train: bool=True
    
