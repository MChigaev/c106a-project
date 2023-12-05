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
from std_msgs.msg import Bool

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

PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHTS_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOPS_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOKS_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEENS_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KINGS_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

def determine_best_move(board, is_white, depth = 3):
    """Given a board, determines the best move.

    Args:
        board (chess.Board): A chess board.
        is_white (bool): Whether the particular move is for white or black.
        depth (int, optional): The number of moves looked ahead.

    Returns:
        chess.Move: The best predicated move.
    """

    best_move = -100000 if is_white else 100000
    best_final = None
    for move in board.legal_moves:
        board.push(move)
        value = _minimax_helper(depth - 1, board, -10000, 10000, not is_white)
        board.pop()
        if (is_white and value > best_move) or (not is_white and value < best_move):
            best_move = value
            best_final = move
    return best_final

def _minimax_helper(depth, board, alpha, beta, is_maximizing):
    if depth <= 0 or board.is_game_over():
        return evaluate(board)

    if is_maximizing:
        best_move = -100000
        for move in board.legal_moves:
            board.push(move)
            value = _minimax_helper(depth - 1, board, alpha, beta, False)
            board.pop()
            best_move = max(best_move, value)
            alpha = max(alpha, best_move)
            if beta <= alpha:
                break
        return best_move
    else:
        best_move = 100000
        for move in board.legal_moves:
            board.push(move)
            value = _minimax_helper(depth - 1, board, alpha, beta, True)
            board.pop()
            best_move = min(best_move, value)
            beta = min(beta, best_move)
            if beta <= alpha:
                break
        return best_move
def evaluate(board):
    """
    Given a particular board, evaluates it and gives it a score.
    A higher score indicates it is better for white.
    A lower score indicates it is better for black.

    Args:
        board (chess.Board): A chess board.

    Returns:
        int: A score indicating the state of the board (higher is good for white, lower is good for black)
    """    

    boardvalue = 0
    
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    material = 100 * (wp - bp) + 300 * (wn - bn) + 300 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)
    
    pawn_sum = sum([PAWN_TABLE[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawn_sum = pawn_sum + sum([-PAWN_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])
    knight_sum = sum([KNIGHTS_TABLE[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knight_sum = knight_sum + sum([-KNIGHTS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishop_sum = sum([BISHOPS_TABLE[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishop_sum = bishop_sum + sum([-BISHOPS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rook_sum = sum([ROOKS_TABLE[i] for i in board.pieces(chess.ROOK, chess.WHITE)]) 
    rook_sum = rook_sum + sum([-ROOKS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.ROOK, chess.BLACK)])
    queens_sum = sum([QUEENS_TABLE[i] for i in board.pieces(chess.QUEEN, chess.WHITE)]) 
    queens_sum = queens_sum + sum([-QUEENS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kings_sum = sum([KINGS_TABLE[i] for i in board.pieces(chess.KING, chess.WHITE)]) 
    kings_sum = kings_sum + sum([-KINGS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.KING, chess.BLACK)])
    
    boardvalue = material + pawn_sum + knight_sum + bishop_sum + rook_sum + queens_sum + kings_sum
    
    return boardvalue

class Controller:
	def __init__(self):
		rospy.init_node('main_execution_loop')
		self.limb = Limb()
		self._tf_buffer = tf2_ros.Buffer()
		self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
		self.right_gripper = robot_gripper.Gripper("right_gripper")
		#self.right_gripper = robot_gripper.Gripper("")
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
		#print(target)
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
			#rospy.logerr("Service call failed: %s" % (e,))
			return

		# Check if result valid, and type of seed ultimately used to get solution
		if (response.result_type[0] > 0):
			#rospy.loginfo("SUCCESS!")
			# Format solution into Limb API-compatible dictionary
			limb_joints = dict(list(zip(response.joints[0].name, response.joints[0].position)))
			#rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
			#rospy.loginfo("------------------")
			#rospy.loginfo("Response Message:\n%s", response)
			#group.plan()
			return limb_joints
		else:
			#rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
			#rospy.logerr("Result Error %d", response.result_type[0])
			return False
	def get_best_angles_from_target_position(self, target, orientation, number_of_trials, frame="right_hand"):
		angle_sets = []
		for i in range(number_of_trials):
			try:
				angle_sets.append(list(self.get_target_angles_from_target_position(target, orientation, frame).values()))
			except:
				continue
		differences = [np.linalg.norm(np.array(angle_sets[i])-np.array(self.get_angles())) for i in range(number_of_trials)]
		return angle_sets[np.argmin(differences)]


	def get_transform(self, target_frame):
		return self.Buffer.lookup_transform("base", target_frame, rospy.Time())
	def move_piece(self, ar_tracker, end_file, end_rank, start_pos, use_derived_position=True):
		offsetx = 0#-0.026924
		offsety = 0#0.026772
		offsetz = 0.04
		transform = self.get_transform(ar_tracker)
		target1 = [transform.transform.translation.x+offsetx, transform.transform.translation.y+offsety, transform.transform.translation.z+0.25]
		if use_derived_position: 
			target1[0] = start_pos[0]
			target1[1] = start_pos[1]
		angles1 = self.get_best_angles_from_target_position(target1, [0, 1, 0, 0], 20)
		self.move(angles1, 0.3)

		target2 = [transform.transform.translation.x+offsetx, transform.transform.translation.y+offsety, transform.transform.translation.z+offsetz]
		angles2 = self.get_best_angles_from_target_position(target2, [0, 1, 0, 0], 20)
		self.move(angles2, 0.3)
		rospy.sleep(1.0)
		self.close()

		self.move(angles1, 0.3)

		x, y = get_square_position(end_file, end_rank)
		z = target1[2]

		target3 = [x+offsetx, y+offsety, z]

		angles3 = self.get_best_angles_from_target_position(target3, [0, 1, 0, 0], 20)
		self.move(angles3, 0.3)

		target4 = [x, y, target2[2]]
		angles4 = self.get_best_angles_from_target_position(target4, [0, 1, 0, 0], 20)
		self.move(angles4, 0.3)

		self.open()

		self.move(angles3, 0.3)

	def move_piece_using_board_pos(self, start_file, start_rank, end_file, end_rank, z):
		offset_z = 0.1
		x1, y1 = get_square_position(start_file, start_rank)
		target1 = [x1, y1, z+0.3]
		angles1 = self.get_best_angles_from_target_position(target1, [0, 1, 0, 0], 20)
		self.move(angles1, 0.1)

		target2 = [x1, y1, z+offset_z]
		angles2 = self.get_best_angles_from_target_position(target2, [0, 1, 0, 0], 20)
		self.move(angles2, 0.1)
		rospy.sleep(1.0)
		self.close()

		self.move(angles1, 0.1)

		x2, y2 = get_square_position(end_file, end_rank)

		target3 = [x2, y2, z+0.3]

		angles3 = self.get_best_angles_from_target_position(target3, [0, 1, 0, 0], 20)
		self.move(angles3, 0.1)

		target4 = [x2, y2, z+offset_z]
		angles4 = self.get_best_angles_from_target_position(target4, [0, 1, 0, 0], 20)
		self.move(angles4, 0.1)

		self.open()
		self.move(angles3, 0.1)

	'''
	def get_all_frames(self):
		return self.Buffer.allFramesAsYAML(rospy.Time())
	'''

def callback(message):
	print(f"Message Received: {message}")
	global blue_check
	blue_check = message
class Dummy():
	def __init__(self):
		self.data = False

if __name__ == "__main__":
	global blue_check 
	blue_check = Dummy()
	control = Controller()
	rospy.Subscriber("camera", Bool, callback)
	rospy.sleep(3)
	#control.move()
	#print(control.get_angles())
	control.open()
	

	#print(control.get_all_frames())

	# view_pos_1 = [0.0173525390625, -1.1609453125, -0.0784716796875, 2.230771484375, -0.0127470703125, -1.08743359375, 0.1408017578125]
	view_pos_1 = [0.048298828125, -1.427421875, -0.174884765625, 2.087876953125, -0.08063671875, -0.6547734375, 0.141345703125]
	# view_pos_1 = [0.6096, 0.1524, 0.508] # x, y, z from base
	# view_pos_2 = [-0.0225810546875, -0.803453125, -0.0276064453125, 1.164185546875, 0.0672490234375, -0.3893779296875, 3.544720703125] # angles
	view_pos_2 = [0.0845869140625, -0.73539453125, -0.3087646484375, 0.8947099609375, 0.1585498046875, -0.1894892578125, 3.543546875]

	corner_32 = [-1.0479443359375, -0.419099609375, 0.052123046875, 1.8389501953125, -1.022732421875, -1.292626953125, 1.2591357421875]

	corner_33 = [-0.4620087890625, 0.58908984375, -1.9538271484375, 0.7064208984375, 2.271958984375, -0.1718671875, 2.0085771484375]

	corner_34 = [0.2188662109375, -0.6931767578125, 0.767427734375, 2.65449609375, 1.1742578125, -2.1683115234375, 0.7951689453125]
	corner_35 = [0.4001396484375, 0.5237109375, -1.3315, 0.4529443359375, 1.3628857421875, -0.6367509765625, 2.43728515625]

	board_view_1 = [-0.0712421875, -0.758689453125, 0.1645068359375, 1.5007255859375, 0.055849609375, -0.7430625, 4.4440576171875]
	board_view_2 = [-0.7769765625, -0.9045869140625, 0.5120439453125, 1.723685546875, -0.385845703125, -0.8170810546875, 4.4440576171875]
	board_view_3 = [-1.056103515625, -1.2960380859375, 0.7043603515625, 2.5631884765625, -0.3878984375, -1.215564453125, 4.4440576171875]

	board_view_4 = [0.0110966796875, -1.06383203125, 0.0953701171875, 2.407669921875, 0.04009765625, -1.3784873046875, 4.44178515625]


	'''control.move(corner_32)
				rospy.sleep(2.0)
				control.move(corner_33)
				rospy.sleep(2.0)
				control.move(corner_34)
				rospy.sleep(2.0)
				control.move(corner_35)
				rospy.sleep(2.0)'''


	# angles1 = control.get_best_angles_from_target_position(view_pos_1, [1, 0, 0, 1.52], 50, "right_hand_camera")
	#control.move(list(.5*np.array(corner_32)+.5*np.array(corner_35)))
	#a=1/0

	def get_square_position(file, rank):
		if file == -1: 
			x = x_max + .09
			y = y_min
			return (x, y)

		x = ((x_max-x_min)/7)*file + x_min
		y = ((y_max-y_min)/7)*rank + y_min #+ 0.0762
		return (x,y)


	def move_and_scan(view_pos, piece_position_tuples_from_based, num_times_scanned, include_corners=False):
			if include_corners: 
				max_index = 36
			else: 
				max_index = 32


			if len(view_pos) == 7:
				control.move(view_pos)
			else:
				angles = control.get_best_angles_from_target_position(view_pos, [0, 0, 1, 0], 50, "right_hand_camera")
				control.move(angles, 0.7)

			rospy.sleep(1.0)
			for i in range(max_index): 
				try: 
					transform = control.get_transform(ar_markers[i])
					piece_transforms[i] = transform

					if num_times_scanned[i] == 0: 
						piece_position_tuples_from_based[i] = (transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z)
					else: 
						piece_position_tuples_from_based[i][0] = (num_times_scanned[i]/(num_times_scanned[i]+1))*piece_position_tuples_from_based[i][0] + (1/(num_times_scanned[i]+1))*transform.transform.translation.x
						piece_position_tuples_from_based[i][1] = (num_times_scanned[i]/(num_times_scanned[i]+1))*piece_position_tuples_from_based[i][1] + (1/(num_times_scanned[i]+1))*transform.transform.translation.y
						piece_position_tuples_from_based[i][2] = (num_times_scanned[i]/(num_times_scanned[i]+1))*piece_position_tuples_from_based[i][2] + (1/(num_times_scanned[i]+1))*transform.transform.translation.Z

					num_times_scanned[i]+=1
				except: 
					continue
			return piece_position_tuples_from_based, num_times_scanned


	global ar_markers
	global piece_names
	global piece_transforms
	global num_times_scanned
	global piece_position_tuples_from_based
	ar_markers = [f"ar_marker_{i}" for i in range(36)]
	piece_names = ["r","n","b","q","k","b","n","r","p","p","p","p","p","p","p","p", "R","N","B","Q","K","B","N","R","P","P","P","P","P","P","P","P", "C1", "C2", "C3", "C4"]

	piece_transforms = [None for i in range(36)]
	num_times_scanned = [0 for i in range(36)]
	piece_position_tuples_from_based = [None for i in range(36)]


	view_positions = [corner_32, corner_33, corner_34, corner_35]

	noise_scale = .1

	noise = np.random.rand(len(view_positions), len(view_positions[0])) * noise_scale 

	second_views = list(noise + np.array(view_positions))
	#view_positions = view_positions + second_views




	for view_pos in view_positions: 
		piece_position_tuples_from_based, num_times_scanned = move_and_scan(view_pos, piece_position_tuples_from_based, num_times_scanned, include_corners=True)


	def move_if_possible():
		global ar_markers
		global piece_names
		global piece_transforms
		global num_times_scanned
		global piece_position_tuples_from_based

		#rospy.Subscriber("camera", Bool, callback)
		print(f"Blue check: {blue_check}")
		if blue_check.data: 
			rospy.sleep(3)
			move_if_possible()


		for i in range(32): 
			piece_transforms[i] = None
			num_times_scanned[i] = 0
			piece_position_tuples_from_based[i] = None

		view_positions = [board_view_1, board_view_2,board_view_3, board_view_4]
		noise_scale = .1

		noise = np.random.rand(len(view_positions), len(view_positions[0])) * noise_scale 

		second_views = list(noise + np.array(view_positions))
		view_positions = view_positions + second_views
		for view_pos in view_positions: 
			piece_position_tuples_from_based, num_times_scanned = move_and_scan(view_pos, piece_position_tuples_from_based, num_times_scanned, include_corners=False)





		print(f"the positions {piece_position_tuples_from_based}")
		print(f"the number of times visited {num_times_scanned}")
		#a = 1/0


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

		global x_min
		x_mins = [value[0] for value in [C1_pos, C3_pos] if value is not None]
		x_min = np.average(x_mins)

		global x_max
		x_maxs = [value[0] for value in [C2_pos, C4_pos] if value is not None]
		x_max = np.average(x_maxs)

		# x_min = min([corner[0] for corner in valid_corners]) +.085 #file a
		# x_min = ((C1_pos[0] + 0.07) + (C3_pos[0] + 0.07))/2
		# x_max = max([corner[0] for corner in valid_corners]) -.085 #file h
		# x_max = C4_pos[0] - 0.07

		# y_min = min([corner[1] for corner in valid_corners]) #rank 1
		# y_min = C1_pos[1] + 0.07
		global y_min
		y_mins = [value[1] for value in [C1_pos, C2_pos] if value is not None]
		y_min = np.average(y_mins) + 0.085
		# y_max = max([corner[1] for corner in valid_corners]) #rank 8
		# y_max = C4_pos[1] - 0.07
		global y_max
		y_maxs = [value[1] for value in [C3_pos, C4_pos] if value is not None]
		y_max = np.average(y_maxs) - 0.085

		global z
		zs = [value[2] for value in [C1_pos, C2_pos, C3_pos, C4_pos] if value is not None]
		z = np.average(zs)

		print((x_min, x_max, y_min, y_max, z))




		def position_file_rank(position): 
			if position == None: 
				return
			x = position[0]
			y = position[1]

			x_board_length = x_max-x_min
			position_along_x_dimension_of_board =x-x_min

			file = int(np.round(np.abs((7*(position_along_x_dimension_of_board)/(x_board_length)))))
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
				rank, file = position_file_rank(piece_position_tuple) 
				rank = int(7 - rank)
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
		computer_is_white = True
		if computer_is_white:
			computer_color = 'w'
		else:
			computer_color = 'b'
		castle_rights = "-"#"KQkq"
		enpassant = "-"
		half_moves = 0 #increment on each black move since the last capture has occurred (needed for 50 half-move stalemate rule)
		full_move_counter = 0 #increment each time computer moves
		
		def make_fenstring(position, computer_color, castle_rights, enpassant, half_moves, full_move_counter): 
			return f"{position} {computer_color} {castle_rights} {enpassant} {str(half_moves)} {str(full_move_counter)}"
		fenstring = make_fenstring(position, computer_color, castle_rights, enpassant, half_moves, full_move_counter)
		print(fenstring)
		#memory = psutil.virtual_memory()
		#print(f"The avaliable memory in bytes is: {memory.avaliable}")
		
		for transform in piece_transforms: 
			del transform
		gc.collect()

		#fenstring = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
		rospy.sleep(1.0)
		board_object = chess.Board(fenstring)
		print("made board")
		result = determine_best_move(board_object, computer_is_white)#chess.engine.SimpleEngine.popen_uci("/home/cc/ee106a/fa23/class/ee106a-aez/c106a-project/workspace/src/chess_solver/src/nodes/stockfish/stockfish-ubuntu-x86-64-avx2")
		#result = engine.play(board)
		best_move = str(result)
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

		start_pos_x, start_pos_y = get_square_position(start_file, start_rank)

		end_file = letter_to_number(best_move[2])
		end_rank = 8-int(best_move[3])

		piece_to_be_moved = board[start_rank][7-start_file]
		piece_to_be_captured = board[end_rank][7-end_file]

		print(f"The piece being moved is {piece_to_be_moved} and its location is {start_file}, {start_rank}")


		if piece_to_be_captured != "": 
			print(f"The piece being captured is {piece_to_be_captured} and its location is {end_file}, {end_rank}")
			control.move_piece_using_board_pos(end_file, end_rank, -1, -1, z+0.10795)

			### LOGIC TO REMOVE THIS PIECE
		control.move_piece_using_board_pos(start_file, start_rank, end_file, end_rank, z+0.10795)
		#control.move_piece(piece_to_be_moved, end_file, end_rank, start_pos=(start_pos_x,start_pos_y))
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


		##Move again 
		rospy.sleep(2)
		move_if_possible()
	move_if_possible()

