#!/usr/bin/env python
from __future__ import print_function
from six.moves import input
import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from geometry_msgs.msg import PoseStamped
from moma_utils.ros.panda import PandaGripperClient
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
import sys
from shapely import Polygon, STRtree, area, contains, buffer, intersection,get_coordinates
from pushing.place_new_obj import place_new_obj_fun,placing_compare_fun
import pickle
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import os
######################################################################################
######################################################################################
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))
######################################################################################
######################## use the PPO model to get the predicted pushing action #######
# def get_x_y(tsdf,occupancy):
#     flag_get = True
#     obs = np.zeros([1,50,50,2])
#     obs[0,:,:,0] = occupancy.copy()
#     obs[0,:,:,1] = tsdf.copy()
#     # checkpoint = './data/model.zip'
#     checkpoint = './data/PPO_model.zip'
#     agent = PPO.load(checkpoint)
#     actions, _ = agent.predict(obs, deterministic=True)
#     obs_tensor = torch.from_numpy(obs)
#     # print(obs_tensor.size())
#     obs_tensor = obs_tensor.permute(0,3,1,2)
#     # print(obs_tensor.size())
#     actions_tensor_tmp =  torch.from_numpy(actions)
#     value,log_prob,entropy = agent.policy.evaluate_actions(obs_tensor,actions_tensor_tmp)
#     # print('value log prob entropy')
#     # print(value,log_prob,entropy)
#     obs_tmp = obs.copy()
#     obs_tensor_tmp = obs_tensor.detach().clone()
#     act_app = np.zeros(len(obs))
#     #print(image_name)  
#     for j in range(3):
#         obs_tmp = np.rot90(obs_tmp,1,(2,1))
#         obs_tmp = obs_tmp.copy()
#         obs_tensor_tmp = obs_tensor_tmp.rot90(1,[3,2])
#         actions_tmp, _ = agent.predict(obs_tmp, deterministic=True)
#         actions_tensor_tmp =  torch.from_numpy(actions_tmp)
#         value_tmp,log_prob_tmp,entropy_tmp = agent.policy.evaluate_actions(obs_tensor_tmp,actions_tensor_tmp)
#         for i in range(len(obs_tensor)):
#             # if float(log_prob_tmp[i])>float(log_prob[i]):
#             if float(value_tmp[i]) > float(value[i]):
#                 actions[i] = actions_tmp[i]
#                 act_app[i] = j * 2.0 +2.0
#                 log_prob[i] = log_prob_tmp[i]
#                 value[i] = value_tmp[i]
#     actions_origin = actions.copy()
#     for _ in range(len(obs)):
#         if act_app[_] == 2:
#             actions[_,0] = 49-actions[_,1]
#             actions[_,1] = actions_origin[_,0]
#         elif act_app[_] == 4:
#             actions[_,0] = 49-actions[_,0]
#             actions[_,1] = 49-actions_origin[_,1]
#         elif act_app[_] == 6:
#             actions[_,0] = actions[_,1]
#             actions[_,1] = 49-actions_origin[_,0]
#     occu = occupancy.copy()
#     if occu[actions.flatten()[0],actions.flatten()[1]] >0:
#        flag_get = False
    
#     print('value: ', value)
#     occu = occu - 100
#     occu[actions.flatten()[0],actions.flatten()[1]] = 255
#     current_time = rospy.Time.now().to_sec()
#     current_time = np.round(current_time)
#     current_time = int(current_time)
#     # image_name = "./data/point_cloud/pushing_result/"+str(current_time)+".png"
#     # cv2.imwrite(image_name,occu)
#     '''
#     for _ in range(len(value)):
#             if float(value[_]) <=-0.1:
#                 act_app[_] = 10
#     '''
#     actions_new = np.c_[actions,act_app.T]   
#     return actions_new,flag_get
##############################################################################
################## switch the controller #####################################
##################TODO: need to call this function every time when moveit or impedance control method is used ###########
def switch_controller(start_controllers, stop_controllers):
    rospy.wait_for_service('/controller_manager/switch_controller')
    try:
        switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        response = switch_controller(start_controllers=start_controllers, stop_controllers=stop_controllers, strictness=SwitchControllerRequest.BEST_EFFORT)
        if response.ok:
            rospy.loginfo("Controllers switched successfully.")
        else:
            rospy.logwarn("Failed to switch controllers: {}".format(response.error_string))
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: {}".format(e))
def switch_to_impedance():
    controller_1 = ["cartesian_impedance_example_controller"]  # Specify the controllers to start
    controller_2 = ["effort_joint_trajectory_controller"]     # Specify the controllers to stop
    start_controller = controller_1
    stop_controller = controller_2
    switch_controller(start_controller, stop_controller)
def switch_to_moveit():
    controller_1 = ["cartesian_impedance_example_controller"]  # Specify the controllers to start
    controller_2 = ["effort_joint_trajectory_controller"]     # Specify the controllers to stop
    start_controller = controller_2
    stop_controller = controller_1
    switch_controller(start_controller, stop_controller)
###############################################################################
####################### impedance control method ##############################
class PerformAction_Impedance(object):
    def __init__(self):
        super(PerformAction_Impedance, self).__init__()
        
        self.rate = rospy.Rate(10)  # 10Hz
    def move(self,x = 0.3,y=0.0,z=0.65):
        i = 0
        self.pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
        while i < 20:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "panda_link0"  # frame ID for the pose
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z
            pose_msg.pose.orientation.x = 1.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z =0.0
            pose_msg.pose.orientation.w = 0.0
            self.pub.publish(pose_msg)
            i +=1
            self.rate.sleep()
#################################################################################
################### moveit control method #######################################
class PerformAction_Moveit(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(PerformAction_Moveit, self).__init__()


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
        print('current pose')
        print(current_pose)
        return current_pose
    def get_gripper_and_joint_pose(self):
        current_pose = self.move_group.get_current_pose().pose
        joint_angle = self.move_group.get_current_joint_values()
        print('current pose:',current_pose)
        print('current joint angles:', joint_angle)
        return current_pose,joint_angle
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
    def move(self,x,y,z):
        move_group = self.move_group
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            # return all_close(pose_goal, current_pose, 0.05)
    def go_to_pose_goal_without_pushing(self,x=0.5,y = 0.0,rotate=0):
        move_group = self.move_group
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = 0.27
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            #return all_close(pose_goal, current_pose, 0.05)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if rotate == 0 or rotate==2:
            if joint_goal[6]< 1.2:
                joint_goal[6]+=tau / 4
            else:
                joint_goal[6]-=tau / 4
            move_group.go(joint_goal, wait=True)
            move_group.stop()
            current_joints = move_group.get_current_joint_values()
            if not all_close(joint_goal, current_joints, 0.05):
                print('ubable to rotate joint 6')
                
        ############################ go down ####################################
        # pose_goal = self.move_group.get_current_pose().pose
        # switch_to_impedance()
        # if pose_goal.position.x <=0.28:

        
        # return all_close(pose_goal, current_pose, 0.05)
    def go_to_pose_goal(self,x=0.5,y = 0.0,rotate=0):
       	switch_to_moveit()
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
        pose_goal.position.z = 0.21
        
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
            #return all_close(pose_goal, current_pose, 0.05)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if rotate == 0 or rotate==2:
            if joint_goal[6]< 1.2:
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
        pose_goal.position.z = 0.2
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
        pose_goal.position.z = 0.149
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
            pose_goal.position.y -=0.1
        elif rotate==1:
            pose_goal.position.x +=0.1
        elif rotate==2:
            pose_goal.position.y +=0.1
        elif rotate==3:
            pose_goal.position.x -=0.1
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)

        move_group.stop()

        move_group.clear_pose_targets()
        ################################ push ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y -=0.1
        elif rotate==1:
            pose_goal.position.x +=0.1
        elif rotate==2:
            pose_goal.position.y +=0.1
        elif rotate==3:
            pose_goal.position.x -=0.1
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
            pose_goal.position.y +=0.06
        elif rotate==1:
            pose_goal.position.x -=0.06
        elif rotate==2:
            pose_goal.position.y -=0.06
        elif rotate==3:
            pose_goal.position.x +=0.06
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
        pose_goal.position.z = 0.4
        
        #gripper_frame = "/panda_hand_tcp"   # Update with camera frame
        #ee_frame = self.eef_link    # Update with world frame

        #(self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        #print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            #return all_close(pose_goal, current_pose, 0.05)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        while rotate>= tau/2:
            rotate = rotate-tau/2
        while rotate<=-tau/2:
            rotate = rotate+tau/2
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
        return all_close(pose_goal, current_pose, 0.05)
    def move_up(self):
        move_group = self.move_group
        pose_goal = move_group.get_current_pose().pose
        pose_goal.position.z = 0.23
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        #print('success: ', success)
        
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
        #print('success: ', success)
        
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
##################################################################################
############## check reaching ####################################################
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
###################################################################################
############################## Image processing ###################################
class DepthImageTransformerTalker:
    def __init__(self):
        #rospy.init_node('pushing_obj', anonymous=True)
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.image_pub = rospy.Publisher('/transformed_depth_image', Image, queue_size=1)
        self.image_pub_occu = rospy.Publisher('/occu_image', Image, queue_size=1)
        self.occu_size = [50,50]
        self.save_path = './data/point_cloud/'
        self.controller_moveit = PerformAction_Moveit()
        self.gripper = PandaGripperClient()
        self.controller_impedance = PerformAction_Impedance()
        # Subscribe to the depth image topic
        #self.puhsing_controller.go_to_obs_joint_state()
        self.controller_moveit.go_to_get_new_obj_pose()
        self.gripper.release(width=0.12)
        self.gripper.grasp(force=30.0)
        rospy.Subscriber('/wrist_camera/depth/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber('/wrist_camera/depth/image_rect_raw', Image, self.depth_image_callback, queue_size=1,buff_size=2**30) ##307200
        #self.debug_pub = rospy.Publisher('/debug_depth', Image, queue_size=1)
        self.time = 0.0
        self.flag_place = True
        self.pre_image = np.zeros([50,50]) 
        self.place_obj_list = []
        self.first_pushing_flag = False
        self.num_obj_placed_by_push = 0
        self.total_push_steps = 0
        self.push_start = 0.76
        self.flag_recover_initial_scene = True
        # file_list = os.listdir("./data/validation_results/real_robot/pushing_compare/")
        idx = 20
        file_name = 'obj_pos'+str(idx)+'.pkl'
        file_path = "./data/validation_results/real_robot/pushing_compare/"+file_name
        f_obj = open(file_path,'rb')
        self.obj_idx = 0
        self.initial_pos=pickle.load(f_obj)
        f_obj.close()                  
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
        points = points[np.where(points[:,0]>-0.56)] ### +y
        points = points[np.where(points[:,1]<0.2)] ###-x
        points = points[np.where(points[:,1]>-0.4)] ### +x
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
        #o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        return pcd
    def image_processing(self,image):
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(image,-1,kernel)
        img_blur=np.where(img_blur <100, 0, img_blur)
        img_blur=np.where(img_blur >=100, 255, img_blur)
        #cv2.imwrite(self.save_path+'occu_filtered_new.png',img_blur)
        current_time = rospy.Time.now().to_sec()
        current_time = np.round(current_time)
        current_time = int(current_time)
        #cv2.imwrite(self.save_path+'occu/'+str(current_time)+'.png',img_blur)
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
        print('obj center',obj_center,aabb_center)
        
        length = aabb.get_extent()
        print('length',length)
        return aabb_center,length,height
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
    def postprocessed_occu_to_tsdf(self,occu):
        Nx,Ny = self.occu_size
        occu_index = np.where(occu>=0.5)
        #print('occu index',occu_index)
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
        #cv2.imwrite(self.save_path+'heightmap.png',heightmap_tmp)
        return heightmap
        
    def decode_image_pos(self):
        pass
    def depth_image_callback(self, msg):
        msg_time = msg.header.stamp.to_sec()
        current_time = rospy.Time.now().to_sec()
        print("Delay is ", current_time - msg_time)
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
                #print(cv_image.shape)
            
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
                
                    pcd_obj = self.depth_pixel_to_camera_obj(cv_image)
                    
                    
                    self.new_obj_center, self.new_obj_length,self.height = self.get_center_new_obj(pcd_obj)
                    w_obj = int(np.ceil(self.new_obj_length[1]*100/2.0))
                    l_obj = int(np.ceil(self.new_obj_length[0]*100/2.0))
                    print('new obj mask length: ',w_obj,l_obj,self.height)
                    self.obj_vertices = np.zeros((4,2))
                    self.obj_vertices[0,0] = -int(w_obj)
                    self.obj_vertices[0,1] = -int(l_obj)
                    self.obj_vertices[1,0] = int(w_obj)
                    self.obj_vertices[1,1] = -int(l_obj)
                    self.obj_vertices[2,0] = int(w_obj)
                    self.obj_vertices[2,1] = int(l_obj)
                    self.obj_vertices[3,0] = -int(w_obj)
                    self.obj_vertices[3,1] = int(l_obj)
                    if w_obj>2 or l_obj>2:
                        self.controller_moveit.go_to_obs_joint_state()
                        self.flag_place = False
                    mask_length = int(max(4*w_obj,4*l_obj))
                    self.new_obj_mask = np.zeros((mask_length,mask_length))
                    self.new_obj_mask[int(mask_length/2)-l_obj:int(mask_length/2)+l_obj,int(mask_length/2)-w_obj:int(mask_length/2)+w_obj] = 1
                    
                    ############################## test
                    # self.gripper.release(width=0.15)
                    # self.controller_moveit.grasp(x=self.new_obj_center[0]+0.01,y=self.new_obj_center[1],z=0.135)
                    # self.gripper.grasp(force=40.0)
                    # self.controller_moveit.move_up()
                    # self.controller_moveit.place(x=0.5,y=0,rotate=0,z=0.3)
                    # self.gripper.release(width=0.15)
                    ###############################
                    # print(self.new_obj_mask)
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
                    #cv2.imwrite(self.save_path+'occu_new.png',occu)
                    #cv2.imwrite(self.save_path+'tsdf_new.png',tsdf)
                    occu = occu/255
                    pre_image = self.pre_image + occu
                    pre_image = pre_image*255/float(np.max(pre_image))
                    pre_image = pre_image.astype(np.uint8)
                    #cv2.imwrite(self.save_path+'difference.png',pre_image)
                    if self.flag_recover_initial_scene:
                        if self.obj_idx>=len(self.initial_pos):
                            flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,self.obj_vertices)
                        else:
                            flag_found = True
                            new_obj_pos = self.initial_pos[self.obj_idx]
                            self.obj_idx += 1
                    else:
                        flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,self.obj_vertices)
                        
                    # flag_found, new_obj_pos = placing_compare_fun(occu,self.new_obj_mask)
                    # if _ is not None:
                    # 	self.pre_image = _.copy()
                    #print('pre max', np.max(self.pre_image))
                    if flag_found:
                        
                        print('place succeed,new item pos',new_obj_pos)
                        #print('vertices: ', new_poly_vetices)
                        #heightmap = self.pcd_to_heightmap(pcd_w)
                        #height = np.mean(heightmap[int(np.min(new_poly_vetices[:,0])):int(np.max(new_poly_vetices[:,0])),int(np.min(new_poly_vetices[:,1])):int(np.max(new_poly_vetices[:,1]))])
                        #print(height)
                        if self.first_pushing_flag:
                            self.num_obj_placed_by_push +=1
                            print('pushing steps and num of obj placed by push',self.total_push_steps,self.num_obj_placed_by_push)
                        self.place_obj_list.append(new_obj_pos)
                        if self.height < 0.164:
                            print('height too low: ', self.height)
                            self.height = 0.164
                        self.gripper.release(width=0.15)
                        self.controller_moveit.grasp(x=self.new_obj_center[0]+0.01,y=self.new_obj_center[1],z=self.height-0.03)
                        self.gripper.grasp(force=40.0)
                        x,y,rot = new_obj_pos
                        if rot <-tau/2:
                           rot = rot + tau/2
                        elif rot >tau/2:
                           rot = rot -tau/2
                        print('place succeed,new item pos',new_obj_pos)
                        action = [x,y,2]
                        action = np.array(action)
                        x,y,_=self.transfrom_action_world(action)
                        #input( "============ Press `Enter` to continue ...") 
                        self.controller_moveit.move_up()
                        print(x,y,rot)
                        flag_place=self.controller_moveit.place(x=x+0.01,y=y,rotate=rot,z=self.height-0.03)
                        #input( "============ Press `Enter` to continue ...") 
                        current_pose = self.controller_moveit.get_gripper_pose
                        print('gripper pose',current_pose)
                        while True:
                            if flag_place:
                                self.gripper.release(width=0.15)
                                break
                            else:
                                self.controller_moveit.move(x=x+0.01,y=y,z=self.height-0.03)
                    

                        self.controller_moveit.move_up()
                        self.controller_moveit.go_to_get_new_obj_pose()
                        self.gripper.grasp()
                        self.flag_place = True
                    else:
                        self.total_push_steps +=1
                        # if not self.first_pushing_flag:
                        #     file_list = os.listdir("./data/validation_results/real_robot/pushing_compare/")
                        #     idx = len(file_list)
                        #     file_name = 'obj_pos'+str(idx)+'.pkl'
                        #     file_path = "./data/validation_results/real_robot/pushing_compare/"+file_name
                        #     f_save = open(file_path,'wb')
                        #     pickle.dump(self.place_obj_list,f_save)
                        #     f_save.close()
                        self.place_obj_list = []
                        self.first_pushing_flag = True
                        # action,flag_get=get_x_y(tsdf_tmp,occu_tmp)
	            
                        # print('push action: ',action)
                        # x,y,rot=self.transfrom_action_world(action)
                        x = self.push_start
                        y = 0.24
                        rot = 0
                        self.controller_moveit.go_to_pose_goal(x=x,y = y,rotate=rot)
                        self.push_start = self.push_start-0.03
                        if self.push_start <=0.32:
                            self.push_start = 0.76
                        
                        # x = x+0.01
                        # if x >0.77:
                        #     x = 0.77
                        # elif x <0.27:
                        #     x = 0.27
                        # if y > 0.25:
                        #     y = 0.25
                        # elif y < -0.25:
                        #     y = -0.25

                        #input( "============ Press `Enter` to continue ...") 
                        # if x>=0.32 and x <=0.76:
                        #     self.controller_moveit.go_to_pose_goal(x=x,y = y,rotate=rot)
                        # elif x >0.76 and abs(y)<0.22:
                        #     self.controller_moveit.go_to_pose_goal(x=x,y = y,rotate=rot)
                        # else:
                        #     if x > 0.76:
                        #         action1.move(x=0.76,y=y,z=0.20)
                        #         # action1.get_gripper_pose()
                        #         switch_to_impedance()
                                
                        #         action2.move(x=x+0.01,y=y,z=0.2)
                        #         # switch_to_moveit()
                        #         # action1.get_gripper_and_joint_pose()
                        #         # switch_to_impedance()
                        #         action2.move(x=x+0.01,y=y,z=0.15)
                        #         # switch_to_moveit()
                        #         # action1.get_gripper_and_joint_pose()
                        #         # switch_to_impedance()
                        #         if rot == 1:
                        #             action2.move(x=x+0.035,y=y,z=0.15)
                        #             action2.move(x=x+0.07,y=y,z=0.15)
                        #         elif rot == 3:
                        #             action2.move(x=x-0.035,y=y,z=0.15)
                        #             action2.move(x=x-0.07,y=y,z=0.15)
                        #         elif rot == 0:
                        #             action2.move(x=x,y=y-0.035,z=0.15)
                        #             action2.move(x=x,y=y-0.07,z=0.15)
                        #         elif rot == 2:
                        #             action2.move(x=x,y=y+0.035,z=0.15)
                        #             action2.move(x=x,y=y+0.07,z=0.15)
                                
                        #         # switch_to_moveit()
                        #         # action1.get_gripper_and_joint_pose()
                        #         # switch_to_impedance()
                        #         action2.move(x=x-0.05,y=y,z=0.2)
                        #         switch_to_moveit()
                        #     elif x<0.32:
                        #         action1.move(x=x,y=y,z=0.3)
                        #         switch_to_impedance()
                        #         z = 0.145
                        #         if y <= 0.1 and y >=-0.1 and x <=0.28:
                        #             x = 0.286
                        #             z = 0.135
                                
                        #         action2.move(x=x,y=y,z=0.25)
                        #         action2.move(x=x,y=y,z=0.19)
                        #         action2.move(x=x,y=y,z=z)
                        #         if rot == 1:
                        #             action2.move(x=x+0.035,y=y,z=z)
                        #             action2.move(x=x+0.07,y=y,z=z)
                        #         elif rot == 3:
                        #             action2.move(x=x-0.035,y=y,z=z)
                        #             action2.move(x=x-0.07,y=y,z=z)
                        #         elif rot == 0:
                        #             action2.move(x=x,y=y-0.035,z=z)
                        #             action2.move(x=x,y=y-0.07,z=z)
                        #         elif rot == 2:
                        #             action2.move(x=x,y=y+0.035,z=z)
                        #             action2.move(x=x,y=y+0.07,z=z)
                        #         switch_to_moveit()
                        #rospy.sleep(3)
    
                        self.controller_moveit.go_to_obs_joint_state()
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("error")
                pass
    def transfrom_action_world(self,action):
        Nx, Ny = self.occu_size
        x,y,rot = action.flatten()
        rot = rot/2
        #print(x,y,rot)
        x = self.min_x+(x-0.5)*(self.max_x-self.min_x)/float(Nx) 
        y = self.min_y+(y-0.5)*(self.max_y-self.min_y)/float(Ny) 
        #print(self.max_x,self.min_x,self.max_y,self.min_y,Nx)
        
        print("action world position: ",x,y,rot)
        return x,y,rot
    def get_occu(self):
        return self.occu
###################################################################################
##################### Main function ###############################################


if __name__ == '__main__':
    rospy.init_node('pushing_obj', anonymous=True)
    '''
    controller_1 = ["cartesian_impedance_example_controller"]  # Specify the controllers to start
    controller_2 = ["effort_joint_trajectory_controller"]     # Specify the controllers to stop
    idx = 0
    '''
    switch_to_moveit()
    action1 = PerformAction_Moveit()
    action1.go_to_obs_joint_state()
    action2 = PerformAction_Impedance()
    ##### if x <=0.28, -0.1<=y<=0.1 ==>> x = 0.29
    ##### unreachable x == 0.77 y>=0.22 or y <= -0.22 z==0.2
    try:
        depth_image_transformer_talker = DepthImageTransformerTalker()
        rospy.spin()
        occu = depth_image_transformer_talker.get_occu()
        print('max occu')
        print(np.max(occu))
        
    except rospy.ROSInterruptException:
        pass
    '''
    try:
        pass
        
        action1.move(x=0.76,y=-0.25,z=0.20)
        action1.get_gripper_pose()
        switch_to_impedance()
        action2.move(x=0.78,y=-0.25,z=0.2)
        switch_to_moveit()
        action1.get_gripper_and_joint_pose()
        switch_to_impedance()
        action2.move(x=0.78,y=-0.25,z=0.15)
        switch_to_moveit()
        action1.get_gripper_and_joint_pose()
        switch_to_impedance()
        action2.move(x=0.74,y=-0.25,z=0.15)
        action2.move(x=0.7,y=-0.25,z=0.15)
        switch_to_moveit()
        action1.get_gripper_and_joint_pose()
        switch_to_impedance()
        action2.move(x=0.78,y=-0.25,z=0.2)

        
        while idx <10:
            y = -0.25 +0.05*idx
            if y <= 0.1 and y >=-0.1:
                x = 0.286
                z = 0.13
            else:
                x = 0.276
                z = 0.12
            action1.move(x=x,y=0.0,z=0.5)
            action1.move(x=x,y=y,z=0.3)
            action1.move(x=x,y=y,z=0.2)
            #action1.move(x=x,y=y,z=0.17)
            action1.get_gripper_pose()
            switch_to_impedance()
            # action2.move(x=x+0.02,y=y,z=0.3)
            # action2.move(x=x+0.02,y=y,z=0.25)
            # action2.move(x=x+0.02,y=y,z=0.2)
            action2.move(x=x,y=y,z=0.19)
            switch_to_moveit()
            action1.get_gripper_and_joint_pose()
            switch_to_impedance()

            x = 0.27+0.02
            action2.move(x=x,y=y,z=z)
            switch_to_moveit()
            action1.get_gripper_pose()
            switch_to_impedance()

            #action2.move(x=0.29,y=0.0,z=0.12)
            action2.move(x=x+0.04,y=y,z=z)
            action2.move(x=x+0.08,y=y,z=z)
            action2.move(x=x+0.07,y=y,z=0.2)
            action2.move(x=x+0.06,y=y,z=0.25)
            switch_to_moveit()
            action1.get_gripper_pose()
            idx += 1
        
        
        while idx<3:
            
            action1.go_to_obs_joint_state()
            action2 = PerformAction_Impedance()
            start_controller = controller_1
            stop_controller = controller_2
            switch_controller(start_controller, stop_controller)
            action2.move(x=0.4)
            start_controller = controller_2
            stop_controller = controller_1
            switch_controller(start_controller, stop_controller)
            idx += 1
        action1.go_to_obs_joint_state()
        
        pass
    except rospy.ROSInterruptException:
        pass
        '''
