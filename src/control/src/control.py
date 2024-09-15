#!/usr/bin/env python3
import rospy
import time
import cv2

class Control_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz

    def run(self):
        while not rospy.is_shutdown() :
            print('control')
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Control_Lancher_Manager()
    test_temp.run()