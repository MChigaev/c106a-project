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
import os 


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

#Chess imports
import chess
import chess.engine
stockfish_path = "stockfish"



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
		wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.3,
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

	'''
	def get_all_frames(self):
		return self.Buffer.allFramesAsYAML(rospy.Time())
	'''



if __name__ == "__main__":
	'''
	control = Controller()
	#control.move()
	#print(control.get_angles())
	control.open()

	#print(control.get_all_frames())

	view_pos_1 = [0.0173525390625, -1.1609453125, -0.0784716796875, 2.230771484375, -0.0127470703125, -1.08743359375, 0.1408017578125]
	view_pos_2 = [-0.0225810546875, -0.803453125, -0.0276064453125, 1.164185546875, 0.0672490234375, -0.3893779296875, 3.544720703125]






	control.move(view_pos_1)
	rospy.sleep(1.5)

	ar_markers = [f"ar_marker_{i}" for i in range(36)]
	piece_names = ["r","n","b","q","k","b","n","r","p","p","p","p","p","p","p","p", "R","N","B","Q","K","B","N","R","P","P","P","P","P","P","P","P", "C1", "C2", "C3", "C4"]
	piece_transforms = []
	piece_position_tuples_from_based = []

	for ar_marker in ar_markers: 
		try: 
			transform = control.get_transform(ar_marker)
			piece_transforms.append(transform)
			piece_position_tuples_from_based.append((transform.transform.translation.x, transform.transform.translation.y))
		except: 
			piece_transforms.append(None)
			piece_position_tuples_from_based.append(None)

	control.move(view_pos_2)
	rospy.sleep(1.5)

	for ar_marker in ar_markers: 
		try: 
			transform = control.get_transform(ar_marker)
			piece_transforms.append(transform)
			piece_position_tuples_from_based.append((transform.transform.translation.x, transform.transform.translation.y))
		except: 
			continue


	board = [["" for i in range(8)] for j in range(8)]

	## get board edges 
	C1_pos = piece_position_tuples_from_based[32]
	C2_pos = piece_position_tuples_from_based[33]
	C3_pos = piece_position_tuples_from_based[34]
	C4_pos = piece_position_tuples_from_based[35]

	valid_corners = [corner for corner in [C1_pos, C2_pos, C3_pos, C4_pos] if corner is not None]

	x_min = min([corner[0] for corner in valid_corners]) +.07 #file a
	x_max = max([corner[0] for corner in valid_corners]) -.07 #file h

	y_min = min([corner[1] for corner in valid_corners]) #rank 1
	y_max = max([corner[1] for corner in valid_corners]) #rank 8

	def position_file_rank(position): 
		if position == None: 
			return
		x = position[0]
		y = position[1]

		file = int(np.round(np.abs((7*(x-x_min)/(x_max-x_min)))))
		rank = int(np.round(np.abs(7*(y-y_min)/(y_max-y_min))))

		file = min(file, 7)
		file = max(file, 0)
		rank = min(rank, 7)
		rank = max(rank, 0)

		return (file, rank)


	##

	for i, piece_position_tuple in enumerate(piece_position_tuples_from_based[:32]): 
		piece_name = piece_names[i]
		try:
			file, rank = position_file_rank(piece_position_tuple) 
		except: 
			continue
		print(f"We are stating that piece {piece_name} is at position {(file, rank)}")
		board[file][rank] = piece_name


	print(board)

	def get_position_string(board):
		board_string = ""
		for row in board: 
			row_string = ""
			num_empty = 0
			for piece in row:
				if piece == "": 
					num_empty+=1
				else: 
					if num_empty != 0: 
						row_string += str(num_empty)
					num_empty = 0
					row_string += piece
			if num_empty != 0: 
				row_string += str(num_empty)
			board_string += "/" + row_string
		return board_string[1:]


	position = get_position_string(board) #"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
	print(f"The board string is {position}")
	computer_color = 'b'
	castle_rights = "KQkq"
	enpassant = "-"
	half_moves = 0 #increment on each black move since the last capture has occurred (needed for 50 half-move stalemate rule)
	full_move_counter = 0 #increment each time computer moves

	def make_fenstring(position, computer_color, castle_rights, enpassant, half_moves, full_move_counter): 
		return f"{position} {computer_color} {castle_rights} {enpassant} {str(half_moves)} {str(full_move_counter)}"

	fenstring = make_fenstring(position, computer_color, castle_rights, enpassant, half_moves, full_move_counter)
	'''
	print(f"current directory: {os.getcwd()}")
	
	fenstring = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 2"
	board = chess.Board(fenstring)
	engine = chess.engine.SimpleEngine.popen_uci(nodes/stockfish/stockfish-windows-x86-64-avx2.exe)
	result = engine.play(board, chess.engine.Limit(time=0.1))
	best_move = result.move
	print("Best move:", best_move)
	
	
	# control.move()
	# transform = control.get_transform("ar_marker_0")
	# target = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
	# joint_angles = list(control.ik_service_client(target).values())
	# print(joint_angles)
	# control.move(joint_angles)
	# control.close()
	# control.move()
	#print(control.ik_service_client())
