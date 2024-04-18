import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

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

if __name__ == "__main__":
    rospy.init_node('controller_switcher_example')
    start_controllers = ["cartesian_impedance_example_controller"]  # Specify the controllers to start
    stop_controllers = ["effort_joint_trajectory_controller"]     # Specify the controllers to stop
    switch_controller(start_controllers, stop_controllers)

