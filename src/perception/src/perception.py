#!/usr/bin/env python3
import cv2
import rospy
from ros_tools import Camera_Image_Listener, LiDAR_PointCloud_Listener, GPS_GNSS_Listener, IMU_Motion_Listener

class Perception_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Sensing_Server', anonymous=False)
        self.rate = rospy.Rate(100) # 10hz
        self.camera_listener = Camera_Image_Listener()
        self.lidar_listener = LiDAR_PointCloud_Listener()
        self.gps_listener = GPS_GNSS_Listener()
        self.imu_listener = IMU_Motion_Listener()

    ### if sensor was shutdown then code will show errors.
    def img_showing(self):
        self.camera_listener.process_msg()
        self.lidar_listener.process_msg()
        self.gps_listener.process_msg()
        self.imu_listener.process_msg()
        
        if self.camera_listener.data_received:
            img = self.camera_listener.img_bgr
            cv2.imshow('window', img)
            cv2.waitKey(1)

        if self.lidar_listener.data_received:
            print(f'here is lidar')

        if self.gps_listener.data_received:
            print(f'here is gps')

        if self.imu_listener.data_received:
            print(f'here is imu')


    def run(self):
        while not rospy.is_shutdown() :
            self.img_showing()
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Perception_Lancher_Manager()
    test_temp.run()