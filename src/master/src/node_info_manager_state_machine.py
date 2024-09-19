#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import psutil
import rospy
import rosnode
import rosnode
from std_msgs.msg import String

class Master_Lancher_Manager:
    def __init__(self) -> None:
        ### No Needs for same name with launch the using name defined by launcher
        rospy.init_node('Master_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz
        self.nodes = rosnode.get_node_names()

        # self.pub_master = rospy.Publisher('/master', String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown() :
            # self.pub_master.publish("hello world")
            print(f'nodes is {self.nodes}')
            # if '/contrl' in self.nodes:
            #     rospy.loginfo("Node A is running")
            #     print('control in master')

            # else:
            #     rospy.loginfo("Node A is not running")
            
            # if '/node_b' in nodes:
            #     rospy.loginfo("Node B is running")
            # else:
            #     rospy.loginfo("Node B is not running")
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Master_Lancher_Manager()
    test_temp.run()


        
        