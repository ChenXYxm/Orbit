#!/usr/bin/env python
from __future__ import print_function
from six.moves import input
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf
import numpy as np
import open3d as o3d
import cv2
from stable_baselines3 import PPO
import torch
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf2_ros
from moma_utils.ros.panda import PandaGripperClient
import sys
#from pushing.place_new_obj import place_new_obj_fun
from shapely import Polygon, STRtree, area, contains, buffer, intersection,get_coordinates
from pushing.place_new_obj import place_new_obj_fun,placing_compare_fun
import pickle

#import matplotlib.pyplot as plt
#import tkinter as tk
#########################################################
#########################################################

def get_pos(new_obj,new_poly_vetices):
    l1 = []
    l2 = []
    pos = [0,0,0]
    obj_l1 = np.array(new_obj[1]) - np.array(new_obj[0])
    obj_l2 = np.array(new_obj[3]) - np.array(new_obj[0])
    sign_obj_p0 = np.sign(np.cross(obj_l1,obj_l2))
    for i in range(2):
        l1.append(np.linalg.norm(new_obj[i]-new_obj[i+1]))
        l2.append(np.linalg.norm(new_poly_vetices[i]-new_poly_vetices[i+1]))
    if l2[0] <1:
        l2[0] = 1
    if l2[1] < 1:
        l2[1] = 1
    if l1[1] < 1:
        l1[1] = 1
    if l1[0] < 1:
        l1[0] = 1
    # print(l1,l2)
    if l1[0] >=l1[1]:
        if l2[0]>=l2[1]:
            
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
    else:
        if l2[0]<l2[1]:
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]

    return pos


##########################################################
##########################################################
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list




class PerformPushing(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(PerformPushing, self).__init__()


        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("perform_pushing", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
	
        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path_pushing",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        
        planning_frame = move_group.get_planning_frame()
        #print("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        #print("============ End effector link: %s" % eef_link)

        # list of all the groups in the robot:
        group_names = robot.get_group_names()
        #print("============ Available Planning Groups:", robot.get_group_names())
          
        #print("============ Printing robot state")
        #print(robot.get_current_state())
        #print("")
        
        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.listener = tf.TransformListener()
        
        ###################### TODO：switch between simulation and real robot
        self.true_eef = "/panda_hand_tcp" ## real
        #self.true_eef = '/panda_hand' ## simulation
        ###################################################################
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        
        # return self.R
    def rotation_matrix_to_quaternion(self,R):
        pass
    def get_gripper_pose(self):
        current_pose = self.move_group.get_current_pose().pose
        # print('current pose')
        # print(current_pose)
    def go_to_joint_state(self):

        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 3
        joint_goal[4] = 0
        joint_goal[5] = tau / 4  # 1/6 of a turn
        joint_goal[6] = tau / 8
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    def go_to_obs_joint_state(self):

        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 10
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 5.2  # 1/6 of a turn
        joint_goal[6] = tau / 8
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    def go_to_get_new_obj_pose(self):
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = 0
        joint_goal[2] = -tau / 9.5
        joint_goal[3] = -tau/4
        joint_goal[4] = 0
        joint_goal[5] = tau/4  # 1/6 of a turn
        joint_goal[6] = 0
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
        ################# adjust the position
        current_joints = move_group.get_current_joint_values()
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.position.x = 0.4  ### 0.74 - 0.32
        pose_goal.position.y = -0.35
        pose_goal.position.z = 0.55
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        #current_joints = move_group.get_current_joint_values()
        #print('current_joints: ',current_joints)
        
    def go_to_pose_goal(self,x=0.5,y = 0.0,rotate=0):
       
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
        ####################################### move to point above target pose ###########################
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = 0.3
        
        #gripper_frame = "/panda_hand_tcp"   # Update with camera frame
        #ee_frame = self.eef_link    # Update with world frame

        #(self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            return all_close(pose_goal, current_pose, 0.05)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if rotate == 0 or rotate==2:
            if joint_goal[6]< 2:
                joint_goal[6]+=tau / 4
            else:
                joint_goal[6]-=tau / 4
            move_group.go(joint_goal, wait=True)
            move_group.stop()
            current_joints = move_group.get_current_joint_values()
            if not all_close(joint_goal, current_joints, 0.05):
                print('ubable to rotate joint 6')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.25
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down1')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.20
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down2')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.15
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down3')
        ################################ push ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y -=0.035
        elif rotate==1:
            pose_goal.position.x +=0.035
        elif rotate==2:
            pose_goal.position.y +=0.035
        elif rotate==3:
            pose_goal.position.x -=0.035
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)

        move_group.stop()

        move_group.clear_pose_targets()
        ################################ push ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y -=0.035
        elif rotate==1:
            pose_goal.position.x +=0.035
        elif rotate==2:
            pose_goal.position.y +=0.035
        elif rotate==3:
            pose_goal.position.x -=0.035
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        print('success: ', success)
        move_group.stop()

        move_group.clear_pose_targets()
        ##########################################################################
        
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: pushing')
        ################################ return ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y +=0.03
        elif rotate==1:
            pose_goal.position.x -=0.03
        elif rotate==2:
            pose_goal.position.y -=0.03
        elif rotate==3:
            pose_goal.position.x +=0.03
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        ############################################
        pose_goal = self.move_group.get_current_pose().pose
        pose_goal.position.z +=0.1
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)

        move_group.stop()

        move_group.clear_pose_targets()
        ##########################################################################
        return all_close(pose_goal, current_pose, 0.05)
    def place(self,x=0.5,y = 0.0,rotate=0,z=0.185):
        

        ###########################################################
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
        ####################################### create middle point
        pose_goal = self.move_group.get_current_pose().pose
        
        pose_goal.position.x = pose_goal.position.x+(x-pose_goal.position.x)/2.
        pose_goal.position.y = pose_goal.position.y+(y-pose_goal.position.y)/2.
        pose_goal.position.z = 0.4
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the middle point')
            # return all_close(pose_goal, current_pose, 0.05)
        # input( "============ complete middle point ...") 
        ####################################### move to point above target pose ###########################
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = 0.4
        
        #gripper_frame = "/panda_hand_tcp"   # Update with camera frame
        #ee_frame = self.eef_link    # Update with world frame

        #(self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            return all_close(pose_goal, current_pose, 0.05)
        # input( "============ complete move above target point ...") 
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if joint_goal[6]+rotate<2.8 and joint_goal[6]+rotate>-2.8:
            joint_goal[6] = joint_goal[6]+rotate
        else:
            joint_goal[6] = joint_goal[6]-tau/2.0+rotate 
        if joint_goal[6]+rotate>2.8:
            joint_goal[6] -= tau/2
        if joint_goal[6]<-2.8:
            joint_goal[6] += tau/2
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        if not all_close(joint_goal, current_joints, 0.05):
            print('ubable to rotate joint 6')
        # input( "============ complete rotation ...") 
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.25
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        # input( "============ complete going down first step ...") 
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        pose_goal.position.z = z
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        # input( "============ complete going down last step ...")
        return all_close(pose_goal, current_pose, 0.05)
    def move_up(self):
        move_group = self.move_group
        pose_goal = move_group.get_current_pose().pose
        pose_goal.position.z = 0.23
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper up after placing')
            
    def grasp(self,x=0.45,y = -0.35,rotate=0,z=0.16):
       
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
        ####################################### move to point above target pose ###########################
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = 0.3
        
        #gripper_frame = "/panda_hand_tcp"   # Update with camera frame
        #ee_frame = self.eef_link    # Update with world frame

        #(self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            return all_close(pose_goal, current_pose, 0.05)
        
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.24
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = z
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        return all_close(pose_goal, current_pose, 0.05)
def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class DepthImageTransformerTalker:
    def __init__(self,placing_method='proposed'):
        rospy.init_node('pushing_obj', anonymous=True)
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.image_pub = rospy.Publisher('/transformed_depth_image', Image, queue_size=1)
        self.image_pub_occu = rospy.Publisher('/occu_image', Image, queue_size=1)
        self.occu_size = [50,50]
        self.save_path = './data/point_cloud/'
        self.puhsing_controller = PerformPushing()
        self.gripper = PandaGripperClient()
        self.placing_method = placing_method
        # Subscribe to the depth image topic
        # self.puhsing_controller.go_to_obs_joint_state()
        self.puhsing_controller.go_to_get_new_obj_pose()
        self.gripper.release(width=0.12)
        self.gripper.grasp(force=30.0)
        rospy.Subscriber('/wrist_camera/depth/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber('/wrist_camera/depth/image_rect_raw', Image, self.depth_image_callback, queue_size=1,buff_size=2**30) ##307200
        #self.debug_pub = rospy.Publisher('/debug_depth', Image, queue_size=1)
        self.time = 0.0
        self.flag_place = True
        self.pre_image = np.zeros([50,50]) 
        
        #rospy.spin()
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        # return self.R
    def depth_pixel_to_camera_obs(self,cv_image):
        [height, width] = cv_image.shape
        
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)        
        x = (u.flatten() - self.camera_cx)/self.camera_fx
        #print(x[:400])
        y = (v.flatten() - self.camera_cy)/self.camera_fy
        
        ##########################################
        ## TODO: switch between real and simulation
        z = cv_image.flatten()/1000 ### real robot
        #z = cv_image.flatten() ### simulation
        ##########################################
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        #print(np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z))
        points=np.zeros((x.shape[0],4))
        points[:,3] = 1
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        points[:,2] = z.flatten()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        points_new_obj = points.copy()
        #o3d.visualization.draw_geometries([pcd])
        '''
        ######################################## drawing
        points = points[np.where(points[:,0]<0.43)]
        points = points[np.where(points[:,0]>-0.3)]
        points = points[np.where(points[:,1]<0.22)]
        points = points[np.where(points[:,1]>-0.32)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot wrist camera##################
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        ############################ before Feb23
        # points = points[np.where(points[:,0]<0.28)]
        # points = points[np.where(points[:,0]>-0.235)]
        # points = points[np.where(points[:,1]<0.207)]
        # points = points[np.where(points[:,1]>-0.290)]
        ##############################################
        # points = points[np.where(points[:,0]<0.35)]
        # points = points[np.where(points[:,0]>-0.305)]
        # points = points[np.where(points[:,1]<0.25)]
        # points = points[np.where(points[:,1]>-0.370)]
        ######################################## after Feb23
        # points = points[np.where(points[:,0]<0.264)]  ### -y
        # points = points[np.where(points[:,0]>-0.225)] ### +y
        # points = points[np.where(points[:,1]<0.198)] ###-x
        # points = points[np.where(points[:,1]>-0.27)] ### +x
        # points = points[np.where(points[:,0]<0.255)]  ### -y
        # points = points[np.where(points[:,0]>-0.22)] ### +y
        # points = points[np.where(points[:,1]<0.2)] ###-x
        # points = points[np.where(points[:,1]>-0.26)] ### +x
        points = points[np.where(points[:,0]<0.45)]  ### -y
        points = points[np.where(points[:,0]>-0.53)] ### +y
        points = points[np.where(points[:,1]<0.22)] ###-x
        points = points[np.where(points[:,1]>-0.4)] ### +x
        # points = points[np.where(points[:,0]<0.255)]  ### -y
        # points = points[np.where(points[:,0]>-0.5)] ### +y
        # points = points[np.where(points[:,1]<0.2)] ###-x
        # points = points[np.where(points[:,1]>-0.35)] ### +x
        ##########################################
        ## TODO: switch between real and simulation
        ############# for simulation #############
        '''
        points = points[np.where(points[:,0]<0.327)]
        points = points[np.where(points[:,0]>-0.173)]
        points = points[np.where(points[:,1]<0.223)]
        points = points[np.where(points[:,1]>-0.277)]
        '''
        ##########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        # o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ##########################################
        ###### get the shape info of new obj #####
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]<-0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]>-0.38)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]<0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]>-0.290)]
        ##########################################
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_new_obj[:,:3])
        o3d.visualization.draw_geometries([pcd_obj])
        ##########################################
        '''
        return pcd, points
    def depth_pixel_to_camera_obj(self,cv_image):
        [height, width] = cv_image.shape
        
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)        
        x = (u.flatten() - self.camera_cx)/self.camera_fx
        #print(x[:400])
        y = (v.flatten() - self.camera_cy)/self.camera_fy
        
        ##########################################
        ## TODO: switch between real and simulation
        z = cv_image.flatten()/1000 ### real robot
        #z = cv_image.flatten() ### simulation
        ##########################################
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        #print(np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z))
        points=np.zeros((x.shape[0],4))
        points[:,3] = 1
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        points[:,2] = z.flatten()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        #o3d.visualization.draw_geometries([pcd])
        '''
        ######################################## drawing
        points = points[np.where(points[:,0]<0.43)]
        points = points[np.where(points[:,0]>-0.3)]
        points = points[np.where(points[:,1]<0.22)]
        points = points[np.where(points[:,1]>-0.32)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot wrist camera##################
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        points = points[np.where(points[:,0]<0.25)]
        points = points[np.where(points[:,0]>0.02)]
        points = points[np.where(points[:,1]<0.18)]
        points = points[np.where(points[:,1]>-0.27)]
        ########################################
        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        # o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        return pcd
    def image_processing(self,image):
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(image,-1,kernel)
        img_blur=np.where(img_blur <100, 0, img_blur)
        img_blur=np.where(img_blur >=100, 255, img_blur)
        # cv2.imwrite(self.save_path+'occu_filtered_new.png',img_blur)
        current_time = rospy.Time.now().to_sec()
        current_time = np.round(current_time)
        current_time = int(current_time)
        # cv2.imwrite(self.save_path+'occu/'+str(current_time)+'.png',img_blur)
        return img_blur
    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.camera_cx = self.camera_matrix[0,2]
        self.camera_cy = self.camera_matrix[1,2]
        self.camera_fx = self.camera_matrix[0,0]
        self.camera_fy = self.camera_matrix[1,1]
        #print('self.camera_cx')
        #print(self.camera_cx)
    def camera_to_world(self,world_points):
    	
        for i in range(len(world_points)):
            world_points[i] = np.dot(self.R,world_points[i])+self.trans
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        plane_model_ori = plane_model
        plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]]).reshape((-1,1))
        world_points=world_points[np.where(world_points[:,0]>=0.275)]
        world_points=world_points[np.where(world_points[:,0]<=0.755)]
        world_points=world_points[np.where(world_points[:,1]<=0.24)]
        world_points=world_points[np.where(world_points[:,1]>=-0.24)]
        
        print('world: ',np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
        self.max_x = np.max(world_points[:,0])
        self.min_x = np.min(world_points[:,0])
        self.max_y = np.max(world_points[:,1])
        self.min_y = np.min(world_points[:,1])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        #o3d.io.write_point_cloud(self.save_path+"pcd_world_frame.ply", pcd)

        ####################### visualization
        obb = pcd.get_oriented_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()
        obb.color = [1,0,0]
        aabb.color = [0,0,0]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd,aabb,axes])
        #o3d.io.write_point_cloud(self.save_path+"pcd_world2.ply", pcd)
        ########################################

        dist_point = (np.dot(world_points,plane_model) + plane_model_ori[3]).reshape(-1)
        inliers = np.array(np.where(dist_point<=0.015)).reshape(-1)
        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        self.pcd = pcd
        self.inlier_cloud = pcd.select_by_index(inliers)
        # Extract outliers (non-plane points)
        self.outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # Visualize the results
        # print('table pcd')
        # o3d.visualization.draw_geometries([self.inlier_cloud])
        # o3d.visualization.draw_geometries([self.outlier_cloud])
        outlier_cloud_np = np.array(self.outlier_cloud.points)
        outlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model_ori
        for i in range(len(outlier_cloud_np)):
            x,y,z = outlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                outlier_cloud_list.append(outlier_cloud_np[i])
        outlier_cloud = o3d.geometry.PointCloud()    
        outlier_cloud_np = np.array(outlier_cloud_list).reshape(-1,3) 
        outlier_cloud_np[:,2] = 0 
        outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_np)
        # o3d.visualization.draw_geometries([outlier_cloud])
        self.outlier_cloud = outlier_cloud
        # Visualize the results
        
        return pcd
    def get_center_new_obj(self,pcd_obj):
        points_np = np.asarray(pcd_obj.points)
        for i in range(len(points_np)):
            points_np[i] = np.dot(self.R,points_np[i])+self.trans
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_np)
        plane_model, inliers = pcd_obj.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        obj_outlier_cloud = pcd_obj.select_by_index(inliers)
  
        obj_inlier_cloud = pcd_obj.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([obj_inlier_cloud])
        # o3d.visualization.draw_geometries([obj_outlier_cloud])
        obj_inlier_cloud_np = np.array(obj_inlier_cloud.points)
        obj_inlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model
        for i in range(len(obj_inlier_cloud_np)):
            x,y,z = obj_inlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                obj_inlier_cloud_list.append(obj_inlier_cloud_np[i])
        obj_inlier_cloud = o3d.geometry.PointCloud()    
        obj_inlier_cloud_np = np.array(obj_inlier_cloud_list).reshape(-1,3) 
        #obj_inlier_cloud_np[:,2] = 0 
        obj_inlier_cloud.points = o3d.utility.Vector3dVector(obj_inlier_cloud_np)
        height = np.mean(obj_inlier_cloud_np[:,2]) #- np.mean(np.array(obj_outlier_cloud.points)[:,2])
        #print('point cloud center')
        aabb = obj_inlier_cloud.get_axis_aligned_bounding_box()
        aabb.color = [0,0,1]
        #o3d.visualization.draw_geometries([obj_inlier_cloud,aabb])
        obj_center = obj_inlier_cloud.get_center()
        aabb_center = aabb.get_center()
        print(obj_center,aabb_center)
        print('length')
        length = aabb.get_extent()
        print(length)
        return aabb_center,length,height,pcd_obj,obj_inlier_cloud
    def pcd_to_tsdf(self):
        Nx, Ny = self.occu_size
        x = np.linspace(self.min_x, self.max_x, Nx)
        y = np.linspace(self.min_y, self.max_y, Ny)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        distance = pts_grid.compute_point_cloud_distance(self.outlier_cloud)
        dist = np.array(distance)
        Tsdf = dist.reshape(Ny,Nx)
        Tsdf = Tsdf*255
        Tsdf = np.fliplr(Tsdf)
        Tsdf = np.rot90(Tsdf)
        #print(np.max(Tsdf),np.min(Tsdf))
        return Tsdf
    def pcd_to_occu(self):
        Nx, Ny = self.occu_size
        pts = np.asarray(self.outlier_cloud.points)
        u = (pts[:,0] - self.min_x)/ ( self.max_x-self.min_x )
        v = (pts[:,1] - self.min_y)/ ( self.max_y-self.min_y )
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
        occupancy = occupancy*255
        occupancy = np.fliplr(occupancy)
        occupancy = np.rot90(occupancy)
        # file_path = "./data/point_cloud/ex_occu.pkl"
        # f_save = open(file_path,'wb')
        # pickle.dump(occupancy,f_save)
        # f_save.close()

        return occupancy
    def pcd_to_occu_obj(self,obj_pcd_w,obj_inlier_cloud):
        obj_points = np.asarray(obj_pcd_w.points)
        print('get new obj occu')
        
        pts = np.asarray(obj_inlier_cloud.points)
        Nx = int(np.ceil((np.max(pts[:,0])-np.min(pts[:,0]))*100))
        Ny = int(np.ceil((np.max(pts[:,1])-np.min(pts[:,1]))*100))
        print(Nx,Ny)
        # o3d.visualization.draw_geometries([obj_pcd_w])
        # o3d.visualization.draw_geometries([obj_inlier_cloud])
        self.max_x_obj = np.max(pts[:,0])
        self.min_x_obj = np.min(pts[:,0])
        self.max_y_obj = np.max(pts[:,1])
        self.min_y_obj = np.min(pts[:,1])
        u = (pts[:,0] - self.min_x_obj)/ ( self.max_x_obj-self.min_x_obj )
        v = (pts[:,1] - self.min_y_obj)/ ( self.max_y_obj-self.min_y_obj )
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
        occupancy = occupancy
        occupancy = np.fliplr(occupancy)
        occupancy = np.rot90(occupancy)
        length = max(Nx,Ny)
        new_obj_occu = np.zeros((int(length/2+Nx+length/2),int(length/2+Ny+length/2)))
        new_obj_occu[int(length/2):int(length/2)+Nx,int(length/2):int(length/2)+Ny] = occupancy
        # cv2.imwrite(self.save_path+'new_occu.png',new_obj_occu*255)
        # file_path = "./data/point_cloud/ex_occu.pkl"
        # f_save = open(file_path,'wb')
        # pickle.dump(occupancy,f_save)
        # f_save.close()

        return new_obj_occu
    def postprocessed_occu_to_tsdf(self,occu):
        Nx,Ny = self.occu_size
        occu_index = np.where(occu>=0.5)
        print('occu index',occu_index)
    def pcd_to_heightmap(self,pcd_w):
        xyz = np.asarray(pcd_w.points)
        
        #print('xyz shape: ',xyz.shape)
        Nx, Ny = self.occu_size
        
        u = (xyz[:,0] - self.min_x)/ ( self.max_x-self.min_x )
        v = (xyz[:,1] - self.min_y)/ ( self.max_y-self.min_y )
        u = (Nx-1)*u
        v = (Ny-1)*v
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        heightmap = np.zeros( (Ny,Nx) )
        xyz[:,0] = u
        xyz[:,1] = v
        for i in range(50):
            for j in range(50):
                tmp_xyz = xyz[np.where(xyz[:,0]==i)].copy()
                tmp_xyz = tmp_xyz[np.where(tmp_xyz[:,1]==j)].copy()
                if len(tmp_xyz)>0:
                    heightmap[i,j] = np.mean(tmp_xyz[:,2])
        
        heightmap_tmp = heightmap.copy()
        heightmap_tmp = (heightmap_tmp-np.min(heightmap_tmp))/(np.max(heightmap_tmp)-np.min(heightmap_tmp))
        heightmap_tmp = heightmap_tmp*255
        # heightmap = heightmap.astype(np.uint8)
        # cv2.imwrite(self.save_path+'heightmap.png',heightmap_tmp)
        return heightmap
        
    def decode_image_pos(self):
        pass
    def depth_image_callback(self, msg):
        msg_time = msg.header.stamp.to_sec()
        current_time = rospy.Time.now().to_sec()
        # print("Delay is ", current_time - msg_time)
        if current_time - msg_time < 2.0:
            try:
                #self.debug_pub.publish(msg)
                # Transform pixel coordinates to camera coordinates
                '''
                print(msg.header.stamp)
                print(msg.header.seq)
                print(msg.header.frame_id)
                '''
                msg_time = msg.header.stamp.to_sec()
                current_time = rospy.Time.now().to_sec()
                print("Delay is ", current_time - msg_time,"current time is: ",current_time)
            
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                # print(cv_image.shape)
            
                #plt.imshow(cv_image)
                ''' debuging
                cv_image_tmp = cv_image.copy()
                cv_image_tmp[np.where(cv_image_tmp<=900)] = 0
                cv_image_tmp[np.where(cv_image_tmp>900)] = 255
                cv_image_tmp = cv_image_tmp.astype(np.uint8)
                cv2.imwrite(self.save_path+'depth.png',cv_image_tmp)
                '''
                camera_frame = "/wrist_camera_depth_optical_frame"  # Update with your camera frame
                world_frame = "/world"    # Update with your world frame

                (self.trans, self.quat) = self.listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
                #print(self.quat,self.trans)
                # get the rotation matrix
                self.quaternion_to_matrix(self.quat)
                if self.flag_place:
                    # self.puhsing_controller.go_to_get_new_obj_pose()
                    pcd_obj = self.depth_pixel_to_camera_obj(cv_image)
                    
                    
                    self.new_obj_center, self.new_obj_length,self.height,pcd_obj_w,obj_inlier_cloud = self.get_center_new_obj(pcd_obj)
                    w_obj = int(np.ceil(self.new_obj_length[1]*100/2.0))
                    l_obj = int(np.ceil(self.new_obj_length[0]*100/2.0))
                    # print('new obj mask length: ',w_obj,l_obj,self.height)
                    if self.placing_method == 'proposed':
                        current_time = rospy.Time.now().to_sec()
                        # print('new obj mask length: ',w_obj,l_obj,self.height)
                        self.obj_vertices = np.zeros((4,2))
                        self.obj_vertices[0,0] = -int(w_obj)
                        self.obj_vertices[0,1] = -int(l_obj)
                        self.obj_vertices[1,0] = int(w_obj)
                        self.obj_vertices[1,1] = -int(l_obj)
                        self.obj_vertices[2,0] = int(w_obj)
                        self.obj_vertices[2,1] = int(l_obj)
                        self.obj_vertices[3,0] = -int(w_obj)
                        self.obj_vertices[3,1] = int(l_obj)
                        new_time = rospy.Time.now().to_sec()
                        delta_time = new_time-current_time
                        self.time += delta_time
                        print('time',self.time)
                        if w_obj>2 or l_obj>2:
                            self.puhsing_controller.go_to_obs_joint_state()
                            self.flag_place = False
                        
                    else:
                        # print('new obj mask length: ',w_obj,l_obj,self.height)
                        # print(2*w_obj,l_obj*2)
                        current_time = rospy.Time.now().to_sec()
                        # self.new_obj_mask = self.pcd_to_occu_obj(pcd_obj_w,obj_inlier_cloud)
                        length_mask = np.max((l_obj*2,w_obj*2))
                        # print('length mask',length_mask)
                        self.new_obj_mask = np.zeros((int(2*length_mask)+14,int(2*length_mask)+14))
                        self.new_obj_mask[length_mask-l_obj+7:length_mask+l_obj+7,length_mask-w_obj+7:length_mask+w_obj+7] = 1
                        new_time = rospy.Time.now().to_sec()
                        delta_time = new_time-current_time
                        print('delta time',delta_time)
                        self.time += delta_time
                        print('time',self.time)
                        if w_obj>2 or l_obj>2:
                            self.puhsing_controller.go_to_obs_joint_state()
                            self.flag_place = False
                # Transform camera coordinates to world coordinates using tf
                else:
                    pcd,points = self.depth_pixel_to_camera_obs(cv_image)
                    world_points = points[:,:3].copy()
                    pcd_w = self.camera_to_world(world_points)
                    occu = self.pcd_to_occu()
                    occu = self.image_processing(occu)
                    occu_tmp = occu.copy()
                    occu = occu.astype(np.uint8)
                    self.occu = occu.copy()
                    tsdf = self.pcd_to_tsdf()
                    tsdf_tmp = tsdf.copy()
                    tsdf = tsdf.astype(np.uint8)
                    # cv2.imwrite(self.save_path+'occu_new.png',occu)
                    # cv2.imwrite(self.save_path+'tsdf_new.png',tsdf)
                    occu = occu/255
                    # pre_image = self.pre_image + occu
                    # pre_image = pre_image*255/float(np.max(pre_image))
                    # pre_image = pre_image.astype(np.uint8)
                    # cv2.imwrite(self.save_path+'difference.png',pre_image)
                    if self.placing_method == 'proposed':
                        current_time = rospy.Time.now().to_sec()
                        flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,self.obj_vertices)
                        new_time = rospy.Time.now().to_sec()
                        delta_time = new_time-current_time
                        self.time += delta_time
                    # if _ is not None:
                    # 	self.pre_image = _.copy()
                    else:
                        current_time = rospy.Time.now().to_sec()
                        flag_found,new_obj_pos=placing_compare_fun(occu,self.new_obj_mask)
                        new_time = rospy.Time.now().to_sec()
                        delta_time = new_time-current_time
                        self.time += delta_time
                    # print('pre max', np.max(self.pre_image))
                    if flag_found:
                        # print('place succeed')
                        # print(new_obj_pos)
                        # print('vertices: ', new_poly_vetices)
                        #heightmap = self.pcd_to_heightmap(pcd_w)
                        #height = np.mean(heightmap[int(np.min(new_poly_vetices[:,0])):int(np.max(new_poly_vetices[:,0])),int(np.min(new_poly_vetices[:,1])):int(np.max(new_poly_vetices[:,1]))])
                        #print(height)
                        self.puhsing_controller.go_to_get_new_obj_pose()
                        if self.height < 0.16:
                            print('height to low: ', self.height)
                            self.height = 0.16
                        self.gripper.release(width=0.15)
                        self.puhsing_controller.grasp(x=self.new_obj_center[0]+0.01,y=self.new_obj_center[1],z=self.height-0.03)
                        self.gripper.grasp(force=40.0)
                        x,y,rot = new_obj_pos
                        if rot <-tau/2:
                           rot = rot + tau
                        elif rot >tau/2:
                           rot = rot -tau
                        action = [x,y,2]
                        action = np.array(action)
                        x,y,_=self.transfrom_action_world(action)
                        # input( "============ Press `Enter` to continue ...") 
                        self.puhsing_controller.move_up()
                        self.puhsing_controller.place(x=x+0.01,y=y,rotate=rot,z=self.height-0.03)
                        # input( "============ Press `Enter` to continue ...") 
                        self.gripper.release(width=0.15)
                        self.puhsing_controller.move_up()
                        self.puhsing_controller.go_to_get_new_obj_pose()
                        self.gripper.grasp(force=30.0)
                        self.flag_place = True
                        print('total time',self.time)
                    else:
                        print('total time',self.time)

                    
                    
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("error")
                pass
    def transfrom_action_world(self,action):
        Nx, Ny = self.occu_size
        x,y,rot = action.flatten()
        rot = rot/2
        print(x,y,rot)
        x = self.min_x+(x-0.5)*(self.max_x-self.min_x)/float(Nx) 
        y = self.min_y+(y-0.5)*(self.max_y-self.min_y)/float(Ny) 
        #print(self.max_x,self.min_x,self.max_y,self.min_y,Nx)
        
        print("action world position: ",x,y,rot)
        return x,y,rot
    def get_occu(self):
        return self.occu
if __name__ == '__main__':
    try:
        depth_image_transformer_talker = DepthImageTransformerTalker(placing_method='other') ##proposed / other
        rospy.spin()
        occu = depth_image_transformer_talker.get_occu()
        # print('max occu')
        # print(np.max(occu))
        
    except rospy.ROSInterruptException:
        pass

