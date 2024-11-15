"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from carla_msgs.msg import CarlaEgoVehicleInfo, CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from transforms3d.euler import quat2euler

import rospy

from std_msgs.msg import Header # pylint: disable=wrong-import-order
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive

from rospy import ServiceException
from rospy.exceptions import ROSInterruptException
    
from src.control.src.carla_sim.carla_spawn_example import spawn_ego_vehicle, clean_ego_vehicle

class Carla_Controller_MPC:
    def __init__(self) -> None:
        self.init_plot = False
        self.wheel_max_angle = None
        self.g_x = None
        self.g_y = None
        self.g_yaw = None
        self.speed = None
        self.get_info = False

    def mpc_for_carla(self,config_file):
        rospy.init_node('asdf')
        self.config_file = config_file
        ## !!! spawn vehicle must be before than subscriber !!!
        spawn_ego_vehicle(config_file)
        
        self.vehicle_control_publisher_arcker = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive , queue_size=1)

        rospy.Subscriber("/carla/ego_vehicle/odometry",Odometry, self.cb_odometry)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info",CarlaEgoVehicleInfo, self.cb_vehicle_info)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_status",CarlaEgoVehicleStatus, self.cb_vehicle_status)
        self.vehicle_control_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

    def cleanup(self,):
        clean_ego_vehicle(self.config_file)

    def cb_odometry(self, msg):
        self.g_x = msg.pose.pose.position.x
        self.g_y = msg.pose.pose.position.y
        _, _, self.g_yaw = quat2euler(
            [msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z])
        # yaw = math.degrees(yaw)

    def cb_vehicle_status(self, msg):
        self.status = msg
        self.get_info = True

    def cb_vehicle_info(self, msg):
        # will not update
        self.wheel_max_angle = msg.wheels[0].max_steer_angle
        print(f'wheel_max_angle is {self.wheel_max_angle}')

    def steering_to_wheel(self, steer_norm, wheel_max_angle):
        wheel_anlge_from_steer = np.abs(steer_norm) * wheel_max_angle
        return wheel_anlge_from_steer if steer_norm >0 else -wheel_anlge_from_steer
