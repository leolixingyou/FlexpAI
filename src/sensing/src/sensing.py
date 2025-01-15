#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from ros_tools import *


class Sensing_Lancher_Manager:
    def __init__(self, platform) -> None:
        rospy.init_node('Sensing_Server', anonymous=False)
        self.rate = rospy.Rate(100) # 10hz

        sensors = SensorConfig(platform)
        self.rgb_camera_listener = Camera_Image_Listener(sensors.sensor_config['Camera'])
        self.seg_camera_listener = Camera_Image_Listener(sensors.sensor_config['Seg_Camera'])
        self.lidar_listener = LiDAR_PointCloud_Listener(sensors.sensor_config['LiDAR'])
        self.gps_listener = GPS_GNSS_Listener(sensors.sensor_config['Gps'])
        self.imu_listener = IMU_Motion_Listener(sensors.sensor_config['Imu'])

        self.baseline_sonsor = None # for adaptive define baseline sensor TODO: Until 2024.10.03. 22:14 no use
        self.sensors_ready = False

    # fusion with camera and LiDAR
    def data_gathering(self,):
        self.rgb_camera_listener.gathering_msg()
        self.lidar_listener.gathering_msg()
        self.gps_listener.gathering_msg()
        self.imu_listener.gathering_msg()

    def publishing(self,):
        # Publish the ego_car([x,y,z], heading, ), obstacles([x,y,z], heading, velocity(state)), lane coordinats...   
        self.gps_listener.pub_gps_msg()


    ### if sensor was shutdown then code will show errors.
    def run(self, tasks):
        while not rospy.is_shutdown() :
            if tasks == 'transfor_cameras':
                self.rgb_camera_listener.transform_to_FLEXPI()
                self.seg_camera_listener.transform_to_FLEXPI()
            # self.data_gathering()
            # self.publishing()
            self.rate.sleep()



if __name__ == "__main__":
    platform = 'carla'
    tasks = 'transfor_cameras'
    test_temp = Sensing_Lancher_Manager(platform)
    test_temp.run(tasks)




