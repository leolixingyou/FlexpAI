#!/usr/bin/env python3
import rospy
import time
import cv2
from std_msgs.msg import String

class Control_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=True)
        self.rate = rospy.Rate(1) # 10hz
        self.pub_master = rospy.Publisher('/control', String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown() :
            print('control')
            self.pub_master.publish("master")
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Control_Lancher_Manager()
    test_temp.run()