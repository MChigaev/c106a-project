#!/usr/bin/env python

"""
Starter script for lab1. 
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ROS imports
import tf
import tf2_ros
import rospy
import baxter_interface
import intera_interface
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from intera_interface import gripper as robot_gripper
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from std_msgs.msg import Header
from moveit_commander import MoveGroupCommander


from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_interface import Limb

NUM_JOINTS = 7

class Controller:
	def __init__(self):
		rospy.init_node('main_execution_loop')
		self.limb = Limb()
		self._tf_buffer = tf2_ros.Buffer()
		self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
		self.right_gripper = robot_gripper.Gripper("right_gripper")
		self.right_gripper.calibrate()
		self.Buffer = tf2_ros.Buffer()
		self.Listener = tf2_ros.TransformListener(self.Buffer)
		rospy.sleep(1.0)
	def move(self, angles = [-0.026388671875, -1.3595009765625, -0.079771484375, 1.3575205078125, 0.0011318359375, 0.01973046875, 1.699166015625]):
		self.traj = MotionTrajectory(limb = self.limb)
		wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.1,
                                         max_joint_accel=0.5)
		waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = self.limb)

		joint_angles = self.limb.joint_ordered_angles()

		waypoint.set_joint_angles(joint_angles = joint_angles)
		self.traj.append_waypoint(waypoint.to_msg())

		waypoint.set_joint_angles(joint_angles = angles)
		self.traj.append_waypoint(waypoint.to_msg())

		result = self.traj.send_trajectory(timeout=None)
		if result is None:
		    rospy.logerr('Trajectory FAILED to send')
		    return

		if result.result:
		    rospy.loginfo('Motion controller successfully finished the trajectory!')
		else:
		    rospy.logerr('Motion controller failed to complete the trajectory with error %s',
		                 result.errorId)
	def get_angles(self):
		return self.limb.joint_ordered_angles()
	# def move_to_frame(target_frame, orientation):
	# 	try:
	# 		pose = self._tf_buffer.lookup_transform(
	# 			self._fixed_frame, self._sensor_frame, rospy.Time())
	# 	except (tf2_ros.LookupException,
    #             tf2_ros.ConnectivityException,
    #             tf2_ros.ExtrapolationException):
    #         # Writes an error message to the ROS log but does not raise an exception
	# 		rospy.logerr("%s: Could not extract pose from TF.", self._name)
	# 		return
	# 	ok = ik_service_client()
	# 	if not ok:
	# 		rospy.logerr("IK Client failed to launch")
	# 		return False
	def open(self):
		self.right_gripper.open()
		rospy.sleep(1.0)
	def close(self):
		self.right_gripper.close()
		rospy.sleep(1.0)

	def ik_service_client(self, target, orientation = [0, 1, 0, 0]):
		group = MoveGroupCommander("right_arm")
		service_name = "ExternalTools/right/PositionKinematicsNode/IKService"
		ik_service_proxy = rospy.ServiceProxy(service_name, SolvePositionIK)
		ik_request = SolvePositionIKRequest()
		header = Header(stamp=rospy.Time.now(), frame_id='base')

		# Create a PoseStamped and specify header (specifying a header is very important!)
		pose_stamped = PoseStamped()
		pose_stamped.header = header

		# Set end effector position: YOUR CODE HERE
		x = float(target[0])
		y = float(target[1])
		z = float(target[2])

		pose_stamped.pose.position.x = x
		pose_stamped.pose.position.y = y
		pose_stamped.pose.position.z = z

		# Set end effector quaternion: YOUR CODE HERE
		pose_stamped.pose.orientation.x = orientation[0]
		pose_stamped.pose.orientation.y = orientation[1]
		pose_stamped.pose.orientation.z = orientation[2]
		pose_stamped.pose.orientation.w = orientation[3]
		group.set_pose_target(pose_stamped)


		# Add desired pose for inverse kinematics
		ik_request.pose_stamp.append(pose_stamped)
		# Request inverse kinematics from base to "right_hand" link
		ik_request.tip_names.append('right_hand')

		rospy.loginfo("Running Simple IK Service Client example.")

		try:
			rospy.wait_for_service(service_name, 5.0)
			response = ik_service_proxy(ik_request)
		except (rospy.ServiceException, rospy.ROSException) as e:
			rospy.logerr("Service call failed: %s" % (e,))
			return

		# Check if result valid, and type of seed ultimately used to get solution
		if (response.result_type[0] > 0):
			rospy.loginfo("SUCCESS!")
			# Format solution into Limb API-compatible dictionary
			limb_joints = dict(list(zip(response.joints[0].name, response.joints[0].position)))
			rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
			rospy.loginfo("------------------")
			rospy.loginfo("Response Message:\n%s", response)
			group.plan()
			return limb_joints
		else:
			rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
			rospy.logerr("Result Error %d", response.result_type[0])
			return False
	def get_transform(self, target_frame):
		return self.Buffer.lookup_transform("base", target_frame, rospy.Time())




if __name__ == "__main__":
	control = Controller()
	control.open()
	# control.move()
	# transform = control.get_transform("ar_marker_0")
	# target = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
	# joint_angles = list(control.ik_service_client(target).values())
	# print(joint_angles)
	# control.move(joint_angles)
	# control.close()
	# control.move()
	#print(control.ik_service_client())
