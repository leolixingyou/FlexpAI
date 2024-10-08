#!/usr/bin/env python3
import numpy as np

import rospy
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from cv_bridge import CvBridge, CvBridgeError
from collections import deque #deque: Append and pop of elements at both ends are overwhelmingly faster.
from carla_msgs.msg import CarlaEgoVehicleStatus

CARLA_SENSOR_CONFIG = {
    "Camera": ['/Ctrl_CV/camera/image', '/Ctrl_CV/perception/camera', Image],
    "LiDAR": ['/Ctrl_CV/lidar/pcd', '/Ctrl_CV/perception/lidar', PointCloud2],
    "Gps": ['/carla/ego_vehicle/gnss', '/Ctrl_CV/perception/gps', NavSatFix],
    "Imu": ['/Ctrl_CV/imu', '/Ctrl_CV/perception/imu', Imu],
}

CARLA_VEHICLE_CONFIG = {
    "Ego": ['/carla/ego_vehicle/vehicle_status', '/Ctrl_CV/control/temp', CarlaEgoVehicleStatus],
}
class Message_Manager:
    def __init__(self) -> None:
        self.msgs = deque(maxlen=30)

    def get_msgs(self, msg):
        self.msgs.append(msg)

class SensorListener:
    def __init__(self, sensor_info):
        sub_topic, pub_topic, msg_type = sensor_info
        rospy.Subscriber(sub_topic, msg_type, self.callback)
        self.pub_msg = rospy.Publisher(pub_topic, msg_type, queue_size=1)
        self.bridge = CvBridge()
        self.msg_manager = Message_Manager()
        self.sensor_data = None
        self.data_received = False
        
    def callback(self, msg):
        self.msg_manager.get_msgs(msg)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [x.data for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True
    


class Camera_Image_Listener(SensorListener):
    def __init__(self, sensor_info):
        super().__init__(sensor_info)

    def msg_to_image(self):
        self.data = self.bridge.imgmsg_to_cv2(self.sensor_data, "bgr8")
        self.data_received = True

    def pub_img(self, image):
        data = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_msg.publish(data)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 0:
            self.datas = [x for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True

class LiDAR_PointCloud_Listener(SensorListener):
    def __init__(self, sensor_info):
        super().__init__(sensor_info)
        
class GPS_GNSS_Listener(SensorListener):
    def __init__(self, sensor_info):
        super().__init__(sensor_info)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [[x.latitude, x.longitude, x.altitude] for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True
    
    def pub_gps_msg(self, ):
        if len(self.msg_manager.msgs) >1:
            self.pub_msg.publish(self.msg_manager.msgs[-1])


class IMU_Motion_Listener(SensorListener):
    def __init__(self,sensor_info):
        super().__init__(sensor_info)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [[x.orientation.x, x.orientation.y, x.orientation.z, x.orientation.w] for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True


class SensorConfig:
    def __init__(self, platform_name) -> None:
        if platform_name == 'carla':
            self.sensor_config = CARLA_SENSOR_CONFIG

class VehicleConfig:
    def __init__(self, platform_name) -> None:
        if platform_name == 'carla':
            self.vehicle_config = CARLA_VEHICLE_CONFIG

class CarlaEgoVehicle_Listener:
    def __init__(self, vehicle_info):
        sub_topic, pub_topic, msg_type = vehicle_info
        rospy.Subscriber(sub_topic, msg_type, self.callback)
        self.pub_msg = rospy.Publisher(pub_topic, msg_type, queue_size=1)
        self.bridge = CvBridge()
        self.msg_manager = Message_Manager()
        self.sensor_data = None
        self.data_received = False

        self.datas = []
        self.times = []
        
    def callback(self, msg):
        self.msg_manager.get_msgs(msg)

    def gathering_msg(self):
        if len(self.msg_manager.msgs) > 1:
            self.datas = [x for x in self.msg_manager.msgs]
            self.times = [x.header.stamp.nsecs for x in self.msg_manager.msgs]
            self.data_received = True



