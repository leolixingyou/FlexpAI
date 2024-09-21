#!/usr/bin/env python3
import rospy
import time
import cv2

class Sensing_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Sensing_Server', anonymous=True)
        self.rate = rospy.Rate(1) # 10hz

    def run(self):
        cnt = 1
        while not rospy.is_shutdown() :
            print('sensing',cnt)
            cnt += 1
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Sensing_Lancher_Manager()
    test_temp.run()