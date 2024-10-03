#!/usr/bin/env python3
import cv2
import numpy as np
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

        self.baseline_sonsor = None # for adaptive define baseline sensor TODO: Until 2024.10.03. 22:14 no use

        self.sensors_ready = False

    ## for visualization test
    def img_showing(self):
        self.camera_listener.msg_to_image()
        self.lidar_listener.gathering_msg()
        self.gps_listener.gathering_msg()
        self.imu_listener.gathering_msg()
        
        if self.camera_listener.data_received:
            img = self.camera_listener.data
            cv2.imshow('window', img)
            cv2.waitKey(1)

        if self.lidar_listener.data_received:
            print(f'here is lidar')

        if self.gps_listener.data_received:
            print(f'here is gps')

        if self.imu_listener.data_received:
            print(f'here is imu')

    # TODO: adaption
    def time_synchronization(self, baseline, todo_list):
        closest_idx = [np.argmin(np.abs(np.array(x) - baseline)) for x in todo_list]
        return closest_idx

    # fusion with camera and LiDAR
    def data_gathering(self,):
        self.camera_listener.gathering_msg()
        self.lidar_listener.gathering_msg()
        self.gps_listener.gathering_msg()
        self.imu_listener.gathering_msg()
        if self.camera_listener.data_received and self.lidar_listener.data_received and self.gps_listener.data_received and self.imu_listener.data_received:
            self.sensors_ready = True

    def synchronization(self,):
        baseline_sensor_times = self.lidar_listener.times[-1]
        need_2_sensor_fusion = [self.camera_listener.times, self.gps_listener.times, self.imu_listener.times]
        return self.time_synchronization(baseline_sensor_times, need_2_sensor_fusion)

    def processing(self, closest_idx):
        baseline_sensor_data = self.lidar_listener[-1]
        camera_data = self.camera_listener.data[closest_idx[0]]
        gps_data = self.gps_listener.data[closest_idx[1]]
        imu_data = self.imu_listener.data[closest_idx[2]]

        # TODO fusion or other process should be added but how to process them parallel?

    def publishing(self,):
        # Publish the ego_car([x,y,z], heading, ), obstacles([x,y,z], heading, velocity(state)), lane coordinats...   
        pass


    ### if sensor was shutdown then code will show errors.
    def run(self):
        while not rospy.is_shutdown() :

            self.data_gathering()
            # TODO: later 2024.10.03.22:56
            # if self.sensors_ready:
            #     closest_idx = self.synchronization()
            #     self.processing(closest_idx)
            #     self.publishing()
            
            if self.sensors_ready:
                1
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Perception_Lancher_Manager()
    test_temp.run()