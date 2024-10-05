#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from ros_tools import SensorListener


class Camera_Image_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/rgb_front/image', '/Ctrl_CV/camera/image', Image, 'camera')
        
    def msg_to_image(self):
        self.data = self.bridge.imgmsg_to_cv2(self.sensor_data, "bgr8")
        self.data_received = True

    def pub_img(self, image):
        data = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_data.publish(data)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [x for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True

class LiDAR_PointCloud_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/lidar', '/Ctrl_CV/lidar/pcd', PointCloud2, 'lidar')
        
class GPS_GNSS_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/gnss', '/Ctrl_CV/gps/gnss', NavSatFix, 'gnss')
    
    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [[x.latitude, x.longitude, x.altitude] for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True
            

class IMU_Motion_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/imu', '/Ctrl_CV/imu', Imu, 'imu')

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [[x.orientation.x, x.orientation.y, x.orientation.z, x.orientation.w] for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True

class Perception_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Preception_Frontend_Server', anonymous=False)
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

    # def processing(self, closest_idx):
    #     self.baseline_sensor_data = self.lidar_listener[-1]
    #     self.camera_listener.data = (self.camera_listener.datas[closest_idx[0]])
    #     self.gps_data = self.gps_listener.datas[closest_idx[1]]
    #     self.imu_data = self.imu_listener.datas[closest_idx[2]]

        # TODO fusion or other process should be added but how to process them parallel?

    def processing(self, closest_idx):
        self.camera_listener.sensor_data = self.camera_listener.datas[closest_idx[0]]
        self.camera_listener.msg_to_image()
        image = self.camera_listener.data
        return image

    def publishing(self,image):
        # Publish the ego_car([x,y,z], heading, ), obstacles([x,y,z], heading, velocity(state)), lane coordinats...   
        self.camera_listener.pub_img(image)


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
                image = self.processing([-1,-1,-1])
                self.publishing(image)
            self.rate.sleep()

if __name__ == "__main__":
    test_temp = Perception_Lancher_Manager()
    test_temp.run()