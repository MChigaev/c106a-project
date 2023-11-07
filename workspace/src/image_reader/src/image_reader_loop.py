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

NUM_JOINTS = 7

def image_read_callback(request):
    pass
def image_reader_server():
    rospy.init_node("image_reader_server")
    rospy.Service('/robot/image_reader_server', ChessBoard, image_read_callback)
    rospy.loginfo('Launching image reader server')
    rospy.spin()

if __name__ == "__main__":
    image_reader_server()