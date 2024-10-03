#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from cv_bridge import CvBridge, CvBridgeError
from collections import deque #deque: Append and pop of elements at both ends are overwhelmingly faster.

class Message_Manager:
    def __init__(self) -> None:
        self.msg_list = deque(maxlen=30)

    def get_msg(self, msg):
        self.msg_list.append(msg)

    def synchronization(self):
        pass


class SensorListener:
    def __init__(self, topic, msg_type, sensor_name):
        self.topic = topic
        self.msg_type = msg_type
        self.sensor_name = sensor_name
        rospy.Subscriber(topic, msg_type, self.callback)
        self.bridge = CvBridge()
        self.msg_manager = Message_Manager()
        self.data_received = False
        
    def callback(self, msg):
        self.msg_manager.get_msg(msg)
        
    def process_msg(self):
        if len(self.msg_manager.msg_list) > 1:
            self.data = self.msg_manager.msg_list[-1]
            self.data_received = True
            
class Camera_Image_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/rgb_front/image', Image, 'camera')
        
    def process_msg(self):
        super().process_msg()
        if len(self.msg_manager.msg_list) > 1:
            self.img_bgr = self.bridge.imgmsg_to_cv2(self.data, "bgr8")
            self.data_received = True
            
class LiDAR_PointCloud_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/lidar', PointCloud2, 'lidar')
        
class GPS_GNSS_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/gnss', NavSatFix, 'gnss')
        
class IMU_Motion_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/imu', Imu, 'imu')