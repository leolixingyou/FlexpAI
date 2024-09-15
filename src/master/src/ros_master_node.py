#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import psutil
import rospy
import rosnode


class Master_Lancher_Manager:
    def __init__(self) -> None:
        ### No Needs for same name with launch the using name defined by launcher
        rospy.init_node('Master_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz

    def run(self):
        while not rospy.is_shutdown() :
            print('master')
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Master_Lancher_Manager()
    test_temp.run()