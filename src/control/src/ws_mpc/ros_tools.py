#!/usr/bin/env python3
import numpy as np
import threading
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from collections import deque #deque: Append and pop of elements at both ends are overwhelmingly faster.

CARLA_SENSOR_CONFIG = {
    "Camera": ['/Ctrl_CV/camera/image', '/Ctrl_CV/perception/camera', Image],
    "LiDAR": ['/Ctrl_CV/lidar/pcd', '/Ctrl_CV/perception/lidar', PointCloud2],
    "Gps": ['/carla/ego_vehicle/gnss', '/Ctrl_CV/perception/gps', NavSatFix],
    "Imu": ['/Ctrl_CV/imu', '/Ctrl_CV/perception/imu', Imu],
    "Odem": ['/carla/ego_vehicle/odometry', '/Ctrl_CV/perception/imu', Odometry],
}

CARLA_VEHICLE_CONFIG = {
    "Ego_Status": ['/carla/ego_vehicle/vehicle_status', '/Ctrl_CV/control/temp', CarlaEgoVehicleStatus],
    "Ego_Control": ['/Ctrl_CV/planning/local_path', '/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl],
}
class Message_Manager:
    def __init__(self) -> None:
        self.msgs = deque(maxlen=30)
        self.lock = threading.Lock()

    def add_message(self, msg):
        with self.lock:
            self.msgs.append(msg)

    def get_msgs(self):
        with self.lock:
            return list(self.msgs)
        
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
        self.msg_manager.add_message(msg)

    def gathering_msg(self):
        self.datas = self.msg_manager.get_msgs()
        self.data_received = len(self.datas) > 0
    


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
        self.pub_msg = None
        self.bridge = CvBridge()
        self.msg_manager = Message_Manager()
        self.sensor_data = None
        self.data_received = False

        self.datas = []
        self.times = []
        
    def callback(self, msg):
        self.msg_manager.add_message(msg)

    def gathering_msg(self):
        self.datas = self.msg_manager.get_msgs()
        self.data_received = len(self.datas) > 0

class Odem_Listener(SensorListener):
    def __init__(self,sensor_info):
        super().__init__(sensor_info)

class Ctrl_CV_CarlaEgoVehicelControl:
    def __init__(self, vehicle_info) -> None:
        sub_topic, pub_topic, msg_type = vehicle_info
        self.pub_msg = rospy.Publisher(pub_topic, msg_type, queue_size=1)

    def pub_control_msg(self, control_data):
        """
        control_data = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'hand_brake': False,
            'reverse': False,
            'gear': 0,
            'manual_gear_shift': False
        }
        
        """
        msg = CarlaEgoVehicleControl()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for key, value in control_data.items():
            setattr(msg, key, value)
        self.pub_msg.publish(msg)