from carla_msgs.msg import CarlaEgoVehicleControl,CarlaEgoVehicleStatus
from std_msgs.msg import Bool
import rospy
from nav_msgs.msg import Odometry

import cv2
import numpy as np
from transforms3d.euler import quat2euler

g_x = 0
g_y = 0
g_z = 0
g_yaw = 0
def cb_odometry(data):
    global g_x,g_y,g_z,g_yaw
    g_x = data.pose.pose.position.x
    g_y = data.pose.pose.position.y
    g_z = data.pose.pose.position.z
    _, _, g_yaw = quat2euler(
        [data.pose.pose.orientation.w,
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z])
    # yaw = math.degrees(yaw)

g_status = CarlaEgoVehicleStatus()
def cb_status(data):
    global g_status
    g_status = data

rospy.init_node('asdf')

_control = CarlaEgoVehicleControl()

vehicle_control_manual_override_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_manual_override", Bool, queue_size=1)
vehicle_control_manual_override_publisher.publish((Bool(True)))

vehicle_control_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

rospy.Subscriber("/carla/ego_vehicle/odometry",Odometry, cb_odometry)
rospy.Subscriber("/carla/ego_vehicle/vehicle_status",CarlaEgoVehicleStatus, cb_status)
while 1:
    _control.throttle = 0.1
    _control.steer = 0.5 ## + is right
    _control.brake = 0
    _control.hand_brake = False
    _control.reverse = False
    _control.gear = 1
    _control.manual_gear_shift = False
    vehicle_control_publisher.publish(_control)

# img = np.zeros([1,1])
# cv2.imshow('img',img)
# while True:
#     key = cv2.waitKey(1)
#     # print(key)
#     if key == ord('w'):
#         _control.gear = 1
#         _control.reverse = _control.gear < 0
#         _control.throttle = 1
#         _control.brake = 0
#     elif key == ord('s'):
#         _control.gear = -1
#         _control.reverse = _control.gear < 0
#         _control.throttle = 1
#         _control.brake = 0
#     elif key == 32:
#         _control.throttle = 0
#         _control.brake = 1
#     vehicle_control_publisher.publish(_control)
#     print(g_x,g_y,g_z,g_yaw)
#     # print(g_status)
#     print(round(g_status.velocity * 3.6, 1),'kph')
# # for i in range(1000,0,-1):
# #     f = i / 100
# #     _control.throttle = max(0,min(1,f))
# #     print(_control.throttle)
# #     vehicle_control_publisher.publish(_control)



