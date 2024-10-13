#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from std_msgs.msg import Header # pylint: disable=wrong-import-order
from ackermann_msgs.msg import AckermannDrive


def get_time():
   return rospy.get_time()

def ros_timestamp(sec=0, nsec=0, from_sec=False):
    if from_sec:
        return rospy.Time.from_sec(sec)
    return rospy.Time(int(sec), int(nsec))

def get_msg_header():
    """
    Get a filled ROS message header
    :return: ROS message header
    :rtype: std_msgs.msg.Header
    """
    header = Header()
    header.frame_id = "map"
    header.stamp = ros_timestamp(sec=get_time(), from_sec=True)
    return header

class Control_Command_Manager:
    def __init__(self) -> None:
        rospy.init_node('Control_Command_Publisher', anonymous=False)
        self.pub_arckermann = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive , queue_size=1)

    def pub_msg(self,):
         
        msg = AckermannDrive()
        msg.steering_angle = self.target_steer
        msg.steering_angle_velocity = 0
        msg.speed = self.target_speed
        msg.acceleration = self.target_acc
        msg.jerk =self.target_jerk 
        self.pub_arckermann.publish(msg)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        temp_flag = False
        img = np.zeros([1,1])
        cv2.imshow('img',img)
        while not rospy.is_shutdown():
            key = cv2.waitKey(1)
            if key == ord('w'):
                self.target_speed = 10
                self.target_steer = 0
                self.target_acc = 1
                self.target_jerk = 0

            elif key == ord('s'):
                self.target_speed = -10
                self.target_steer = 0
                self.target_acc = 1
                self.target_jerk = 0
            elif key == 32:
                self.target_speed = 0
                self.target_steer = 0
                self.target_acc = 0
                self.target_jerk = 0
            elif key == ord('q'):
                break

            if key != -1:
                self.pub_msg()
            rate.sleep()

if __name__ == "__main__":
    test_temp = Control_Command_Manager()
    test_temp.run()