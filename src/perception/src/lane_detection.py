#!/usr/bin/env python3
import numpy as np

import rospy
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from cv_bridge import CvBridge, CvBridgeError
from collections import deque #deque: Append and pop of elements at both ends are overwhelmingly faster.
from ros_tools import SensorListener


class Camera_Image_Listener(SensorListener):
    def __init__(self):
        super().__init__('/carla/ego_vehicle/rgb_front/image', Image, 'camera')
        
    def msg_to_image(self):
        if len(self.msg_manager.msgs) > 1:
            self.data = self.msg_manager.msgs[-1]
            self.data = self.bridge.imgmsg_to_cv2(self.data, "bgr8")
            self.data_received = True

# TODO: Plugin YOLOvP
class LaneDetection:
    def __init__(self) -> None:
        rospy.init_node('Lane_Detection', anonymous=False)

    