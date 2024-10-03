#!/usr/bin/env python3
import numpy as np

import rospy
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from cv_bridge import CvBridge, CvBridgeError
from collections import deque #deque: Append and pop of elements at both ends are overwhelmingly faster.

class Message_Manager:
    def __init__(self) -> None:
        self.msgs = deque(maxlen=30)

    def get_msgs(self, msg):
        self.msgs.append(msg)

class SensorListener:
    def __init__(self, sub_topic, pub_topic, msg_type, sensor_name):
        rospy.Subscriber(sub_topic, msg_type, self.callback)
        self.pub_data = rospy.Publisher(pub_topic, msg_type, queue_size=1)
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
            
