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
import psutil
import gc


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
	def move(self, angles = [-0.026388671875, -1.3595009765625, -0.079771484375, 1.3575205078125, 0.0011318359375, 0.01973046875, 1.699166015625], max_speed_ratio=0.3):
		self.traj = MotionTrajectory(limb = self.limb)
		wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=max_speed_ratio,
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

	def get_target_angles_from_target_position(self, target, orientation, frame):
		#group = MoveGroupCommander("right_arm")
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
		#group.set_pose_target(pose_stamped)


		# Add desired pose for inverse kinematics
		ik_request.pose_stamp.append(pose_stamped)
		# Request inverse kinematics from base to "right_hand" link
		ik_request.tip_names.append(frame)

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
			#group.plan()
			return limb_joints
		else:
			rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
			rospy.logerr("Result Error %d", response.result_type[0])
			return False
	def get_best_angles_from_target_position(self, target, orientation, number_of_trials, frame="right_hand"):
		angle_sets = []
		for i in range(number_of_trials):
			angle_sets.append(list(self.get_target_angles_from_target_position(target, orientation, frame).values()))
		differences = [np.linalg.norm(np.array(angle_sets[i])-np.array(self.get_angles())) for i in range(number_of_trials)]
		return angle_sets[np.argmin(differences)]


	def get_transform(self, target_frame):
		return self.Buffer.lookup_transform("base", target_frame, rospy.Time())
	def move_piece(self, ar_tracker, end_file, end_rank):
		offsetx = 0#-0.026924
		offsety = 0#0.026772
		offsetz = 0.04
		transform = self.get_transform(ar_tracker)
		target1 = [transform.transform.translation.x+offsetx, transform.transform.translation.y+offsety, transform.transform.translation.z+0.25]
		angles1 = self.get_best_angles_from_target_position(target1, [0, 1, 0, 0], 20)
		self.move(angles1, 0.1)

		target2 = [transform.transform.translation.x+offsetx, transform.transform.translation.y+offsety, transform.transform.translation.z+offsetz]
		angles2 = self.get_best_angles_from_target_position(target2, [0, 1, 0, 0], 20)
		self.move(angles2, 0.1)
		rospy.sleep(1.0)
		self.close()

		self.move(angles1, 0.1)

		x, y = get_square_position(end_file, end_rank)
		z = target1[2]

		target3 = [x+offsetx, y+offsety, z]

		angles3 = self.get_best_angles_from_target_position(target3, [0, 1, 0, 0], 20)
		self.move(angles3, 0.1)

		target4 = [x, y, target2[2]]
		angles4 = self.get_best_angles_from_target_position(target4, [0, 1, 0, 0], 20)
		self.move(angles4, 0.1)

		self.open()

		self.move(angles3, 0.1)


	'''
	def get_all_frames(self):
		return self.Buffer.allFramesAsYAML(rospy.Time())
	'''



if __name__ == "__main__":
	
	control = Controller()
	#control.move()
	# print(control.get_angles())
	control.open()

	#print(control.get_all_frames())

	# view_pos_1 = [0.0173525390625, -1.1609453125, -0.0784716796875, 2.230771484375, -0.0127470703125, -1.08743359375, 0.1408017578125]
	view_pos_1 = [0.048298828125, -1.427421875, -0.174884765625, 2.087876953125, -0.08063671875, -0.6547734375, 0.141345703125]
	# view_pos_1 = [0.6096, 0.1524, 0.508] # x, y, z from base
	# view_pos_2 = [-0.0225810546875, -0.803453125, -0.0276064453125, 1.164185546875, 0.0672490234375, -0.3893779296875, 3.544720703125] # angles
	view_pos_2 = [0.0845869140625, -0.73539453125, -0.3087646484375, 0.8947099609375, 0.1585498046875, -0.1894892578125, 3.543546875]







	control.move(view_pos_1)
	# angles1 = control.get_best_angles_from_target_position(view_pos_1, [1, 0, 0, 1.52], 50, "right_hand_camera")
	# control.move(angles1)
	rospy.sleep(2.0)

	ar_markers = [f"ar_marker_{i}" for i in range(36)]
	piece_names = ["r","n","b","q","k","b","n","r","p","p","p","p","p","p","p","p", "R","N","B","Q","K","B","N","R","P","P","P","P","P","P","P","P", "C1", "C2", "C3", "C4"]
	piece_transforms = [None for i in range(36)]
	piece_position_tuples_from_based = [None for i in range(36)]

	for i in range(36): 
		try: 
			transform = control.get_transform(ar_markers[i])
			piece_transforms[i] = transform
			piece_position_tuples_from_based[i] = (transform.transform.translation.x, transform.transform.translation.y)
		except: 
			continue

	control.move(view_pos_2)
	rospy.sleep(3.0)

	for i in range(36):
		try: 
			transform = control.get_transform(ar_markers[i])
			piece_transforms[i] = transform
			piece_position_tuples_from_based[i] = (transform.transform.translation.x, transform.transform.translation.y)
		except: 
			continue

	board = [["" for i in range(8)] for j in range(8)]
	board_ar_trackers = [["" for i in range(8)] for j in range(8)]
	board_transforms = [[None for i in range(8)] for j in range(8)]

	## get board edges 
	C1_pos = piece_position_tuples_from_based[32]
	C2_pos = piece_position_tuples_from_based[33]
	C3_pos = piece_position_tuples_from_based[34]
	C4_pos = piece_position_tuples_from_based[35]
	print((C1_pos, C2_pos, C3_pos, C4_pos))


	# valid_corners = [corner for corner in [C1_pos, C2_pos, C3_pos, C4_pos] if corner is not None]

	x_mins = [value[0] for value in [C1_pos, C3_pos] if value is not None]
	x_min = np.min(x_mins)

	x_maxs = [value[0] for value in [C2_pos, C4_pos] if value is not None]
	x_max = np.min(x_maxs)

	# x_min = min([corner[0] for corner in valid_corners]) +.07 #file a
	# x_min = ((C1_pos[0] + 0.07) + (C3_pos[0] + 0.07))/2
	# x_max = max([corner[0] for corner in valid_corners]) -.07 #file h
	# x_max = C4_pos[0] - 0.07

	# y_min = min([corner[1] for corner in valid_corners]) #rank 1
	# y_min = C1_pos[1] + 0.07
	y_mins = [value[1] for value in [C1_pos, C2_pos] if value is not None]
	y_min = np.min(y_mins)
	# y_max = max([corner[1] for corner in valid_corners]) #rank 8
	# y_max = C4_pos[1] - 0.07
	y_maxs = [value[1] for value in [C3_pos, C4_pos] if value is not None]
	y_max = np.min(y_maxs)

	print((x_min, x_max, y_min, y_max))

	def get_square_position(file, rank):
		x = ((x_max-x_min)/8)*file + x_min
		y = ((y_max-y_min)/8)*rank + y_min + 0.00762
		return (x,y)


	def position_file_rank(position): 
		if position == None: 
			return
		x = position[0]
		y = position[1]

		file = int(np.round(np.abs((7*(x-x_min)/(x_max-x_min)))))
		rank = int(np.round(np.abs(7*(y-y_min)/(y_max-y_min)))) 

		print(7*(x-x_min)/(x_max-x_min))
		print(7*(y-y_min)/(y_max-y_min))
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
		board_ar_trackers[file][rank] = ar_markers[i]
		board_transforms[file][rank] = piece_transforms[i]


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
	print(f"current directory: {os.getcwd()}")
	#memory = psutil.virtual_memory()
	#print(f"The avaliable memory in bytes is: {memory.avaliable}")
	
	for transform in piece_transforms: 
		del transform
	gc.collect()

	fenstring = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 2"
	rospy.sleep(1.0)
	#board = chess.Board(fenstring)
	#engine = chess.engine.SimpleEngine.popen_uci("nodes/stockfish/stockfish-ubuntu-x86-64-avx2")
	#result = engine.play(board, chess.engine.Limit(time=1.0))
	best_move = "d2g5"#str(result.move)
	print("Best move:", best_move)

	def letter_to_number(letter):
		if letter == "a":
			return 7
		elif letter == "b":
			return 6
		elif letter == "c":
			return 5
		elif letter == "d": 
			return 4
		elif letter == "e": 
			return 3
		elif letter == "f": 
			return 2
		elif letter == "g": 
			return 1
		elif letter == "h": 
			return 0

	start_file = letter_to_number(best_move[0])
	start_rank = 8-int(best_move[1])

	end_file = letter_to_number(best_move[2])
	end_rank = 8-int(best_move[3])

	piece_to_be_moved = board_ar_trackers[start_file][start_rank]

	piece_to_be_captured = board_ar_trackers[end_file][end_rank]

	print(f"The piece being moved is {piece_to_be_moved} and its location is {start_file}, {start_rank}")


	if piece_to_be_captured != "": 
		print(f"The piece being captured is {piece_to_be_captured} and its location is {end_file}, {end_rank}")

		### LOGIC TO REMOVE THIS PIECE
	control.move_piece(piece_to_be_moved, end_file, end_rank)
	## Logiv to pick up piece and put it in the new position 
	
	
	# control.move()
	# transform = control.get_transform("ar_marker_0")
	# target = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
	# joint_angles = list(control.ik_service_client(target).values())
	# print(joint_angles)
	# control.move(joint_angles)
	# control.close()
	# control.move()
	#print(control.ik_service_client())
