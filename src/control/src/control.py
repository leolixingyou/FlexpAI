#!/usr/bin/env python3
import rospy
import time
import cv2
from carla_msgs.msg import CarlaEgoVehicleControl

class Control_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=True)
        self.rate = rospy.Rate(1) # 10hz
        self.pub_twist = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=10)

    
    def pub_control(self,):
        msg = CarlaEgoVehicleControl()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.pub_twist.publish(msg)

    def run(self):
        cnt = 1
        while not rospy.is_shutdown() :
            print('control',cnt)
            cnt += 1
            self.pub_control()
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Control_Lancher_Manager()
    test_temp.run()