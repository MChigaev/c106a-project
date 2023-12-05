#!/usr/bin/env python

from safety_loop import open_camera, release_camera, safety_cam
import rospy
from std_msgs.msg import Bool


def main_loop():

    cap = open_camera()

    if cap is None:
        print("Error: Could not open camera.")
        return

    try:
        safety_cam(cap, pub)

    finally:
        release_camera(cap)

if __name__ == "__main__":
    rospy.init_node("blue_detector", anonymous=True)
    pub = rospy.Publisher("camera", Bool, queue_size=10)
    main_loop()
